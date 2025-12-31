// fluidos.cpp
// Compilar:
// g++ -O3 -march=native -fopenmp fluidos.cpp `pkg-config --cflags --libs opencv4` -o fluidos
//
// Archivos esperados en la MISMA carpeta:
// - Ubuntu-B.ttf
// - frases.txt   (una frase por línea)
//
// Requisitos extra:
// - python3
// - Pillow (PIL):  python3 -m pip install pillow
//
// Qué hace este fichero (según tu petición):
// 1) En cada segmento (y también al inicio), elige una frase aleatoria de frases.txt
// 2) Genera un colisionador de texto con Pillow (usando Ubuntu-B.ttf), ajustando el tamaño
//    para que el texto ocupe ~80% del ancho de la simulación (NX), con límite de altura
// 3) Carga esa imagen generada y la usa como máscara de sólido
// 4) El nombre del vídeo incluye: timestamp + epoch + frase

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================
// Configuración
// ============================================================
static constexpr int W = 1920, H = 1080;

static constexpr int CELL = 2;
static constexpr int NX = W / CELL;
static constexpr int NY = H / CELL;

static constexpr float DT = 1.0f / 60.0f;

static constexpr float VISC = 5e-4f;
static constexpr float DIFF_MASA  = 2e-4f;  // difusión de la masa (tinte)
static constexpr int   JACOBI_ITERS = 40;

static constexpr float WATER_FILL = 1.0f;
static constexpr int   WATERLINE_Y = int(NY * (1.0f - WATER_FILL));

// Nota: DENSIDAD_PINTURA > 1 hunde; <=1 no hunde.
static constexpr float DENSIDAD_PINTURA = 0.05f;
static constexpr float GRAVEDAD = 1.0f;      // 0 desactiva
static constexpr float DECAY_MASA = 0.0005f;
static constexpr float MASA_MAX = 8.0f;

// Vorticidad
static constexpr float VORT_EPS = 1.0f;
static constexpr float VORT_SMOOTH_SIGMA = 2.0f;
static constexpr int   VORT_SMOOTH_K_MULT = 6;

// Emisores
static constexpr int   RADIO_EMISOR = 40;
static constexpr float EMISOR_RATE_MASA = 0.35f;
static constexpr float EMISOR_VELOCIDAD = 3.5f;
static constexpr float EMISOR_SPAWN_SEGUNDOS = 4.0f;
static constexpr float EMISOR_VIDA_SEGUNDOS  = 4.0f;

// Fondo RGB (0..1)
static constexpr float FONDO_R = 1.0f;
static constexpr float FONDO_G = 1.0f;
static constexpr float FONDO_B = 1.0f;

// Resistencia a mezclar color (0..1)
static constexpr float RESISTENCIA_MEZCLA = 0.35f;
static constexpr float DIFF_COLOR = DIFF_MASA * (1.0f - RESISTENCIA_MEZCLA);

// ---- Vídeo
static constexpr bool  GUARDAR_VIDEO = true;
static constexpr int   VIDEO_FPS = int(std::lround(1.0 / DT));
static constexpr const char* VIDEO_CODEC = "mp4v";

// Rotar vídeo cada N segundos Y reiniciar simulación
static constexpr float DURACION_VIDEO_SEGUNDOS = 600.0f;
static constexpr int   DURACION_VIDEO_FRAMES =
    int(std::lround(DURACION_VIDEO_SEGUNDOS * float(VIDEO_FPS)));

// UI (para priorizar simulación)
static constexpr int   MOSTRAR_CADA_N_FRAMES = 5;
static constexpr int   IMPRIMIR_CADA_FRAMES = 300;

// ------------------------------------------------------------
// Colisionador de texto (Pillow)
// ------------------------------------------------------------
static constexpr const char* TTF_FONT   = "Ubuntu-B.ttf";
static constexpr const char* FRASES_TXT = "frases.txt";

// Ajuste para que el texto ocupe ~80% del ancho del collider (NX)
static constexpr float TEXTO_ANCHO_OBJETIVO = 0.80f; // 80% del ancho (NX)
// Límite de altura para que no se dispare en frases cortas
static constexpr float TEXTO_ALTO_MAX       = 0.70f; // 70% del alto (NY)

static constexpr int   UMBRAL_ALPHA_COLISIONADOR = 10; // 0..255

// Guarda el PNG generado por Pillow (por debug). Si false, se borra el PNG temporal.
static constexpr bool  GUARDAR_DEBUG_PNG_COLISIONADOR = false;

// ============================================================
// Utilidades
// ============================================================
static inline float clamp01(float x) { return std::min(1.0f, std::max(0.0f, x)); }
static inline int idx(int x, int y) { return y * NX + x; }

static std::string timestamp_ahora() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return std::string(buf);
}

static long long epoch_ahora() {
    using namespace std::chrono;
    return (long long)duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

static inline std::string trim_copy(std::string s) {
    auto notspace = [](unsigned char c){ return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notspace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notspace).base(), s.end());
    return s;
}

static std::vector<std::string> leer_frases(const std::string& path) {
    std::ifstream f(path);
    std::vector<std::string> lines;
    if (!f.is_open()) return lines;
    std::string line;
    while (std::getline(f, line)) {
        line = trim_copy(line);
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

// Sanitiza una frase para nombre de archivo: [a-zA-Z0-9_-] y recorta
static std::string slug_frase(const std::string& s, size_t maxlen = 60) {
    std::string out;
    out.reserve(std::min(maxlen, s.size()));
    for (unsigned char c : s) {
        if (std::isalnum(c)) out.push_back((char)c);
        else if (c==' ' || c=='-' || c=='_' || c=='.') out.push_back('_');
        // ignorar otros
        if (out.size() >= maxlen) break;
    }
    // compactar múltiples '_'
    std::string compact;
    compact.reserve(out.size());
    bool prev_us = false;
    for (char c : out) {
        if (c=='_') {
            if (!prev_us) compact.push_back(c);
            prev_us = true;
        } else {
            compact.push_back(c);
            prev_us = false;
        }
    }
    // trim '_' inicial/final
    while (!compact.empty() && compact.front()=='_') compact.erase(compact.begin());
    while (!compact.empty() && compact.back()=='_') compact.pop_back();
    if (compact.empty()) compact = "frase";
    return compact;
}

// ============================================================
// VideoWriter con nombre: timestamp + epoch + frase
// ============================================================
static cv::VideoWriter abrir_nuevo_writer(const std::string& prefijo,
                                         const std::string& frase,
                                         std::string& nombre_salida) {
    const std::string ts = timestamp_ahora();
    const long long ep = epoch_ahora();
    const std::string slug = slug_frase(frase);

    nombre_salida = prefijo + "_" + ts + "_ep" + std::to_string(ep) + "_" + slug + ".mp4";

    int fourcc = cv::VideoWriter::fourcc(VIDEO_CODEC[0], VIDEO_CODEC[1], VIDEO_CODEC[2], VIDEO_CODEC[3]);
    cv::VideoWriter w(nombre_salida, fourcc, VIDEO_FPS, cv::Size(W, H), true);
    if (!w.isOpened()) {
        throw std::runtime_error("No se pudo abrir VideoWriter. Prueba otro FOURCC o instala codecs.");
    }
    return w;
}

static void imprimir_stats_segmento(int frame_i,
                                    const std::chrono::steady_clock::time_point& inicio,
                                    const std::string& nombre) {
    using namespace std::chrono;
    auto ahora = steady_clock::now();
    double elapsed = duration<double>(ahora - inicio).count();
    double fps_real = (elapsed > 1e-9) ? (double(frame_i) / elapsed) : 0.0;

    auto fmt = [](double sec)->std::string{
        if (!(sec == sec) || sec < 0) return "??:??:??";
        int s = int(std::lround(sec));
        int h = s / 3600;
        int m = (s % 3600) / 60;
        int ss = s % 60;
        char b[32];
        std::snprintf(b, sizeof(b), "%02d:%02d:%02d", h, m, ss);
        return std::string(b);
    };

    std::cout << "[" << nombre << "] "
              << "frames: " << frame_i
              << " | tiempo real: " << fmt(elapsed)
              << " | fps real: " << cv::format("%.2f", fps_real)
              << "\n" << std::flush;
}

// ============================================================
// Campo 2D float
// ============================================================
struct Campo {
    std::vector<float> a;
    Campo() : a(NX * NY, 0.0f) {}
    explicit Campo(float v) : a(NX * NY, v) {}

    inline float& at(int x, int y) { return a[idx(x,y)]; }
    inline float  at(int x, int y) const { return a[idx(x,y)]; }

    void fill(float v) { std::fill(a.begin(), a.end(), v); }

    void mul_inplace(const Campo& m) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)a.size(); ++i) a[i] *= m.a[i];
    }
};

struct Campo3 {
    std::vector<cv::Vec3f> a;
    Campo3() : a(NX * NY, cv::Vec3f(0,0,0)) {}
    inline cv::Vec3f& at(int x, int y) { return a[idx(x,y)]; }
    inline cv::Vec3f  at(int x, int y) const { return a[idx(x,y)]; }

    void fill(const cv::Vec3f& v) { std::fill(a.begin(), a.end(), v); }
};

// ============================================================
// Generar colisionador de texto con Python+Pillow
// - Genera un PNG temporal (L o RGBA)
// - Lo carga con OpenCV
// - Devuelve Campo solido (1 sólido, 0 vacío)
// ============================================================
static bool escribir_archivo(const std::string& path, const std::string& content) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.write(content.data(), (std::streamsize)content.size());
    return true;
}

static int ejecutar(const std::string& cmd) {
    // std::system devuelve implementación-dependiente; 0 suele ser ok.
    return std::system(cmd.c_str());
}

static Campo mascara_colisionador_texto_pillow(const std::string& frase) {
    // Temp paths (en la carpeta actual para evitar problemas de permisos)
    const long long ep = epoch_ahora();
    const std::string slug = slug_frase(frase, 40);

    const std::string tmp_py  = "tmp_render_text_" + std::to_string(ep) + "_" + slug + ".py";
    const std::string tmp_png = "tmp_collider_"     + std::to_string(ep) + "_" + slug + ".png";

    // Script Python (Pillow)
    // - Crea imagen L (grayscale) Nx x Ny
    // - Calcula tamaño de fuente por búsqueda binaria para encajar 80% del ancho (y <= 70% del alto)
    // - Centra el texto
    std::ostringstream py;
    py <<
R"(import sys
from PIL import Image, ImageDraw, ImageFont

W = int(sys.argv[1])
H = int(sys.argv[2])
font_path = sys.argv[3]
out_path = sys.argv[4]
phrase = sys.argv[5]

target_w = float(sys.argv[6]) * W
max_h    = float(sys.argv[7]) * H

img = Image.new("L", (W, H), 0)
draw = ImageDraw.Draw(img)

def measure(font):
    # bbox -> (x0,y0,x1,y1)
    bbox = draw.textbbox((0,0), phrase, font=font)
    return (bbox[2]-bbox[0], bbox[3]-bbox[1], bbox)

# Búsqueda binaria de font size
lo, hi = 6, 600
best = lo
while lo <= hi:
    mid = (lo + hi) // 2
    try:
        font = ImageFont.truetype(font_path, mid)
    except Exception:
        # Si falla el TTF, abortamos con error
        raise
    w, h, bbox = measure(font)
    if w <= 0 or h <= 0:
        lo = mid + 1
        continue
    if w <= target_w and h <= max_h:
        best = mid
        lo = mid + 1
    else:
        hi = mid - 1

font = ImageFont.truetype(font_path, best)
w, h, bbox = measure(font)

# Centrado (teniendo en cuenta bbox)
x0, y0, x1, y1 = bbox
text_w = x1 - x0
text_h = y1 - y0

x = (W - text_w) // 2 - x0
y = (H - text_h) // 2 - y0

draw.text((x, y), phrase, fill=255, font=font)

img.save(out_path)
)";

    if (!escribir_archivo(tmp_py, py.str())) {
        throw std::runtime_error("No se pudo escribir el script temporal de Python: " + tmp_py);
    }

    // Ejecutar python
    // Nota: pasamos la frase como argumento (puede contener espacios)
    std::ostringstream cmd;
    cmd << "python3 "
        << "\"" << tmp_py << "\" "
        << NX << " " << NY << " "
        << "\"" << TTF_FONT << "\" "
        << "\"" << tmp_png << "\" "
        << "\"" << frase << "\" "
        << TEXTO_ANCHO_OBJETIVO << " " << TEXTO_ALTO_MAX;

    int rc = ejecutar(cmd.str());

    // Limpieza del .py
    std::error_code ec;
    std::filesystem::remove(tmp_py, ec);

    if (rc != 0) {
        // Intentar borrar PNG si existe
        std::filesystem::remove(tmp_png, ec);
        throw std::runtime_error("Falló la generación del colisionador con Pillow. "
                                 "Comprueba python3 y `pip install pillow`, y que exista Ubuntu-B.ttf.");
    }

    // Cargar PNG
    cv::Mat img = cv::imread(tmp_png, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::filesystem::remove(tmp_png, ec);
        throw std::runtime_error("No se pudo leer el PNG generado: " + tmp_png);
    }

    // Si no queremos debug, borramos
    if (!GUARDAR_DEBUG_PNG_COLISIONADOR) {
        std::filesystem::remove(tmp_png, ec);
    } else {
        std::cout << "[debug] collider guardado: " << tmp_png << "\n";
    }

    cv::Mat alpha;
    if (img.channels() == 1) {
        alpha = img;
    } else if (img.channels() == 4) {
        std::vector<cv::Mat> ch;
        cv::split(img, ch);
        alpha = ch[3];
    } else {
        // Si vino RGB, lo pasamos a gris
        cv::cvtColor(img, alpha, cv::COLOR_BGR2GRAY);
    }

    if (alpha.rows != NY || alpha.cols != NX) {
        cv::resize(alpha, alpha, cv::Size(NX, NY), 0, 0, cv::INTER_AREA);
    }

    Campo solido(0.0f);
    for (int y = 0; y < NY; ++y) {
        const uchar* row = alpha.ptr<uchar>(y);
        for (int x = 0; x < NX; ++x) {
            solido.at(x, y) = (row[x] > UMBRAL_ALPHA_COLISIONADOR) ? 1.0f : 0.0f;
        }
    }
    return solido;
}

// ============================================================
// Derivadas con máscara
// ============================================================
static Campo divergencia_mascara(const Campo& u, const Campo& v, const Campo& m) {
    Campo div(0.0f);
    #pragma omp parallel for schedule(static)
    for (int y = 1; y < NY - 1; ++y) {
        for (int x = 1; x < NX - 1; ++x) {
            float mC = m.at(x,y);
            if (mC == 0.0f) { div.at(x,y) = 0.0f; continue; }

            float mR = m.at(x+1,y);
            float mL = m.at(x-1,y);
            float mD = m.at(x,y+1);
            float mU = m.at(x,y-1);

            float uR = u.at(x+1,y) * mR;
            float uL = u.at(x-1,y) * mL;
            float vD = v.at(x,y+1) * mD;
            float vU = v.at(x,y-1) * mU;

            div.at(x,y) = 0.5f * ((uR - uL) + (vD - vU));
        }
    }
    div.mul_inplace(m);
    return div;
}

static Campo gradx_mascara(const Campo& p, const Campo& m) {
    Campo gx(0.0f);
    #pragma omp parallel for schedule(static)
    for (int y = 1; y < NY - 1; ++y) {
        for (int x = 1; x < NX - 1; ++x) {
            if (m.at(x,y) == 0.0f) { gx.at(x,y) = 0.0f; continue; }
            float mR = m.at(x+1,y);
            float pR = p.at(x+1,y) * mR;
            float mL = m.at(x-1,y);
            float pL = p.at(x-1,y) * mL;
            gx.at(x,y) = 0.5f * (pR - pL);
        }
    }
    gx.mul_inplace(m);
    return gx;
}

static Campo grady_mascara(const Campo& p, const Campo& m) {
    Campo gy(0.0f);
    #pragma omp parallel for schedule(static)
    for (int y = 1; y < NY - 1; ++y) {
        for (int x = 1; x < NX - 1; ++x) {
            if (m.at(x,y) == 0.0f) { gy.at(x,y) = 0.0f; continue; }
            float mD = m.at(x,y+1);
            float pD = p.at(x,y+1) * mD;
            float mU = m.at(x,y-1);
            float pU = p.at(x,y-1) * mU;
            gy.at(x,y) = 0.5f * (pD - pU);
        }
    }
    gy.mul_inplace(m);
    return gy;
}

// ============================================================
// Condiciones de contorno: paredes cerradas + máscara
// ============================================================
static void aplicar_bordes_vel(Campo& u, Campo& v, const Campo& m) {
    #pragma omp parallel for schedule(static)
    for (int x = 0; x < NX; ++x) {
        u.at(x,0)=0; u.at(x,NY-1)=0;
        v.at(x,0)=0; v.at(x,NY-1)=0;
    }
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < NY; ++y) {
        u.at(0,y)=0; u.at(NX-1,y)=0;
        v.at(0,y)=0; v.at(NX-1,y)=0;
    }
    u.mul_inplace(m);
    v.mul_inplace(m);
}

// ============================================================
// Advección bilinear (escalar)
// ============================================================
static Campo advectar_escalar(const Campo& q, const Campo& u, const Campo& v, float dt, const Campo& m) {
    Campo out(0.0f);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if (m.at(x,y) == 0.0f) { out.at(x,y)=0.0f; continue; }

            float xf = float(x) - dt * u.at(x,y);
            float yf = float(y) - dt * v.at(x,y);

            xf = std::min(float(NX-1) - 1e-3f, std::max(0.0f, xf));
            yf = std::min(float(NY-1) - 1e-3f, std::max(0.0f, yf));

            int x0 = int(std::floor(xf));
            int y0 = int(std::floor(yf));
            int x1 = std::min(NX-1, x0+1);
            int y1 = std::min(NY-1, y0+1);

            float sx = xf - float(x0);
            float sy = yf - float(y0);

            float q00 = q.at(x0,y0);
            float q10 = q.at(x1,y0);
            float q01 = q.at(x0,y1);
            float q11 = q.at(x1,y1);

            float q0 = q00*(1-sx) + q10*sx;
            float q1 = q01*(1-sx) + q11*sx;

            out.at(x,y) = q0*(1-sy) + q1*sy;
        }
    }
    out.mul_inplace(m);
    return out;
}

// ============================================================
// Difusión Jacobi con máscara (escalar)
// ============================================================
static Campo difundir_jacobi_mascara(const Campo& x_in, float diff, float dt, int iters, const Campo& m) {
    const float a = diff * dt;
    Campo x = x_in;
    Campo x0 = x_in;

    #pragma omp parallel for schedule(static)
    for (int xk = 0; xk < NX; ++xk) { x.at(xk,0)=0; x.at(xk,NY-1)=0; }
    #pragma omp parallel for schedule(static)
    for (int yk = 0; yk < NY; ++yk) { x.at(0,yk)=0; x.at(NX-1,yk)=0; }
    x.mul_inplace(m);

    Campo x_new(0.0f);

    for (int it = 0; it < iters; ++it) {
        #pragma omp parallel for schedule(static)
        for (int y = 1; y < NY - 1; ++y) {
            for (int xk = 1; xk < NX - 1; ++xk) {
                if (m.at(xk,y) == 0.0f) { x_new.at(xk,y) = 0.0f; continue; }

                float mU = m.at(xk,y-1);
                float mD = m.at(xk,y+1);
                float mL = m.at(xk-1,y);
                float mR = m.at(xk+1,y);

                float nb_sum = x.at(xk,y-1)*mU + x.at(xk,y+1)*mD + x.at(xk-1,y)*mL + x.at(xk+1,y)*mR;
                float cnt = std::max(1.0f, mU + mD + mL + mR);

                x_new.at(xk,y) = (x0.at(xk,y) + a * nb_sum) / (1.0f + a * cnt);
            }
        }

        #pragma omp parallel for schedule(static)
        for (int xk = 0; xk < NX; ++xk) { x_new.at(xk,0)=0; x_new.at(xk,NY-1)=0; }
        #pragma omp parallel for schedule(static)
        for (int yk = 0; yk < NY; ++yk) { x_new.at(0,yk)=0; x_new.at(NX-1,yk)=0; }

        x_new.mul_inplace(m);
        x.a.swap(x_new.a);
    }
    return x;
}

// ============================================================
// Proyección (incompresibilidad)
// ============================================================
static void proyectar(Campo& u, Campo& v, int iters, const Campo& m) {
    Campo div = divergencia_mascara(u, v, m);
    Campo p(0.0f), p_new(0.0f);

    for (int it = 0; it < iters; ++it) {
        #pragma omp parallel for schedule(static)
        for (int y = 1; y < NY - 1; ++y) {
            for (int x = 1; x < NX - 1; ++x) {
                if (m.at(x,y) == 0.0f) { p_new.at(x,y) = 0.0f; continue; }

                float mU = m.at(x,y-1);
                float mD = m.at(x,y+1);
                float mL = m.at(x-1,y);
                float mR = m.at(x+1,y);

                float nb_sum = p.at(x,y-1)*mU + p.at(x,y+1)*mD + p.at(x-1,y)*mL + p.at(x+1,y)*mR;
                float cnt = std::max(1.0f, mU + mD + mL + mR);

                p_new.at(x,y) = (nb_sum - div.at(x,y)) / cnt;
            }
        }

        #pragma omp parallel for schedule(static)
        for (int xk = 0; xk < NX; ++xk) { p_new.at(xk,0)=0; p_new.at(xk,NY-1)=0; }
        #pragma omp parallel for schedule(static)
        for (int yk = 0; yk < NY; ++yk) { p_new.at(0,yk)=0; p_new.at(NX-1,yk)=0; }

        p_new.mul_inplace(m);
        p.a.swap(p_new.a);
    }

    Campo gx = gradx_mascara(p, m);
    Campo gy = grady_mascara(p, m);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < NX*NY; ++i) {
        u.a[i] = (u.a[i] - gx.a[i]) * m.a[i];
        v.a[i] = (v.a[i] - gy.a[i]) * m.a[i];
    }
}

// ============================================================
// Vorticidad (confinement)
// ============================================================
static void confinamiento_vorticidad(Campo& u, Campo& v, float eps, const Campo& m) {
    Campo dv_dx = gradx_mascara(v, m);
    Campo du_dy = grady_mascara(u, m);
    Campo w(0.0f);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < NX*NY; ++i) w.a[i] = (dv_dx.a[i] - du_dy.a[i]) * m.a[i];

    Campo absw(0.0f);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < NX*NY; ++i) absw.a[i] = std::fabs(w.a[i]);

    Campo absw_s = absw;
    if (VORT_SMOOTH_SIGMA > 0.0f) {
        int k = int(std::max(3.0f, std::round(VORT_SMOOTH_SIGMA * float(VORT_SMOOTH_K_MULT))));
        if (k % 2 == 0) k += 1;

        cv::Mat mat(NY, NX, CV_32F, absw_s.a.data());
        cv::GaussianBlur(mat, mat, cv::Size(k,k), VORT_SMOOTH_SIGMA, VORT_SMOOTH_SIGMA, cv::BORDER_REPLICATE);
    }

    Campo gx = gradx_mascara(absw_s, m);
    Campo gy = grady_mascara(absw_s, m);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if (m.at(x,y) == 0.0f) continue;

            float gxx = gx.at(x,y);
            float gyy = gy.at(x,y);
            float mag = std::sqrt(gxx*gxx + gyy*gyy) + 1e-6f;

            float nx = gxx / mag;
            float ny = gyy / mag;

            float ww = w.at(x,y);

            u.at(x,y) += (ny * ww) * eps;
            v.at(x,y) += (-nx * ww) * eps;
        }
    }

    aplicar_bordes_vel(u, v, m);
}

// ============================================================
// Emisores
// ============================================================
struct Emisor {
    int x=0, y=0;
    float vx=0, vy=0;
    int r=RADIO_EMISOR;
    float rate=EMISOR_RATE_MASA;
    cv::Vec3f color{1,1,1};
    int ttl_frames=0;
    int edad_frames=0;
    std::vector<uint8_t> mask;  // 0/1 por celda
};

static cv::Vec3f color_brillante_aleatorio(std::mt19937& rng) {
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    cv::Vec3f c(d(rng), d(rng), d(rng));
    return cv::Vec3f(0.20f + 0.80f*c[0], 0.20f + 0.80f*c[1], 0.20f + 0.80f*c[2]);
}

static Emisor crear_emisor_aleatorio(std::mt19937& rng, const Campo& m) {
    std::uniform_int_distribution<int> edgeDist(0,3);
    int borde = edgeDist(rng);

    int margen = RADIO_EMISOR + 2;
    std::uniform_int_distribution<int> xDist(margen, NX-1-margen);
    std::uniform_int_distribution<int> yDist(std::max(margen, WATERLINE_Y+2), NY-1-margen);

    Emisor em;
    em.ttl_frames = int(std::lround(EMISOR_VIDA_SEGUNDOS / DT));
    em.color = color_brillante_aleatorio(rng);

    if (borde == 0) { // arriba -> abajo
        em.x = xDist(rng);
        em.y = std::max(margen, WATERLINE_Y + 2);
        em.vx = 0; em.vy = +EMISOR_VELOCIDAD;
    } else if (borde == 1) { // derecha -> izquierda
        em.x = NX-1-margen;
        em.y = yDist(rng);
        em.vx = -EMISOR_VELOCIDAD; em.vy = 0;
    } else if (borde == 2) { // abajo -> arriba
        em.x = xDist(rng);
        em.y = NY-1-margen;
        em.vx = 0; em.vy = -EMISOR_VELOCIDAD;
    } else { // izquierda -> derecha
        em.x = margen;
        em.y = yDist(rng);
        em.vx = +EMISOR_VELOCIDAD; em.vy = 0;
    }

    em.mask.assign(NX*NY, 0);

    int r2 = em.r * em.r;
    for (int yy = std::max(0, em.y - em.r); yy <= std::min(NY-1, em.y + em.r); ++yy) {
        for (int xx = std::max(0, em.x - em.r); xx <= std::min(NX-1, em.x + em.r); ++xx) {
            int dx = xx - em.x;
            int dy = yy - em.y;
            if (dx*dx + dy*dy <= r2) {
                if (m.at(xx,yy) > 0.5f) em.mask[idx(xx,yy)] = 1;
            }
        }
    }

    return em;
}

// ============================================================
// Reset simulación
// ============================================================
static void resetear_simulacion(Campo& u, Campo& v,
                                Campo& masa,
                                Campo3& absorcion_premul,
                                std::vector<Emisor>& emisores,
                                int& cuenta_spawn, int intervalo_spawn_frames) {
    u.fill(0.0f);
    v.fill(0.0f);
    masa.fill(0.0f);
    absorcion_premul.fill(cv::Vec3f(0,0,0));
    emisores.clear();
    cuenta_spawn = intervalo_spawn_frames;
}

// ============================================================
// Main
// ============================================================
int main() {
    cv::setUseOptimized(true);
    cv::setNumThreads(1);

#ifdef _OPENMP
    omp_set_dynamic(0);
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    omp_set_num_threads((int)hw);
#endif

    // Comprobaciones mínimas de ficheros
    if (!std::filesystem::exists(TTF_FONT)) {
        std::cerr << "[error] No existe " << TTF_FONT << " en la carpeta actual.\n";
        return 1;
    }
    if (!std::filesystem::exists(FRASES_TXT)) {
        std::cerr << "[error] No existe " << FRASES_TXT << " en la carpeta actual.\n";
        return 1;
    }

    // RNG por ejecución
    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());

    // Leer frases
    auto frases = leer_frases(FRASES_TXT);
    if (frases.empty()) {
        std::cerr << "[error] No hay frases en " << FRASES_TXT << " (o no se pudo leer)\n";
        return 1;
    }
    std::uniform_int_distribution<int> pick(0, (int)frases.size()-1);

    // Máscara base de agua
    Campo base_liquido(0.0f);
    for (int y = WATERLINE_Y; y < NY; ++y)
        for (int x = 0; x < NX; ++x)
            base_liquido.at(x,y) = 1.0f;

    // Estado sim
    Campo u(0.0f), v(0.0f);
    Campo masa(0.0f);
    Campo3 absorcion_premul;
    std::vector<Emisor> emisores;

    int intervalo_spawn_frames = std::max(1, int(std::lround(EMISOR_SPAWN_SEGUNDOS / DT)));
    int cuenta_spawn = intervalo_spawn_frames;

    // Ventana
    cv::namedWindow("Fluido", cv::WINDOW_NORMAL);
    cv::resizeWindow("Fluido", W, H);

    // Video
    cv::VideoWriter writer;
    std::string nombre_video;

    auto inicio_segmento = std::chrono::steady_clock::now();
    int frame_en_segmento = 0;

    cv::Mat small(NY, NX, CV_8UC3);
    cv::Mat shown;

    // Fondo
    const cv::Vec3f fondo_rgb(FONDO_R, FONDO_G, FONDO_B);

    // Máscaras (m: líquido activo; solido: colisionador)
    Campo solido(0.0f);
    Campo m(0.0f);

    std::string frase_actual;

    auto iniciar_nuevo_segmento = [&]() {
        if (GUARDAR_VIDEO && writer.isOpened()) {
            writer.release();
            std::cout << "[video] cerrado: " << nombre_video << "\n" << std::flush;
        }

        // Escoger frase y generar colisionador con Pillow
        frase_actual = frases[pick(rng)];
        std::cout << "[texto] frase elegida: " << frase_actual << "\n";

        solido = mascara_colisionador_texto_pillow(frase_actual);

        // m = agua - solido
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < NX*NY; ++i) {
            float vv = base_liquido.a[i] * (1.0f - solido.a[i]);
            m.a[i] = (vv > 0.5f) ? 1.0f : 0.0f;
        }

        resetear_simulacion(u, v, masa, absorcion_premul, emisores, cuenta_spawn, intervalo_spawn_frames);

        if (GUARDAR_VIDEO) {
            writer = abrir_nuevo_writer("stable_fluids_paint", frase_actual, nombre_video);
            std::cout << "[video] abierto: " << nombre_video << "\n" << std::flush;
        }

        inicio_segmento = std::chrono::steady_clock::now();
        frame_en_segmento = 0;
    };

    // Start
    try {
        iniciar_nuevo_segmento();
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << "\n";
        return 1;
    }

    while (true) {
        if (DURACION_VIDEO_FRAMES > 0 && frame_en_segmento >= DURACION_VIDEO_FRAMES) {
            try {
                iniciar_nuevo_segmento();
            } catch (const std::exception& e) {
                std::cerr << "[error] " << e.what() << "\n";
                break;
            }
        }

        // spawn emisores
        cuenta_spawn -= 1;
        if (cuenta_spawn <= 0) {
            emisores.push_back(crear_emisor_aleatorio(rng, m));
            cuenta_spawn = intervalo_spawn_frames;
        }

        // inyección
        std::vector<Emisor> vivos;
        vivos.reserve(emisores.size());

        for (auto& em : emisores) {
            em.edad_frames += 1;
            if (em.edad_frames <= em.ttl_frames) {
                const cv::Vec3f abs_color(
                    1.0f - clamp01(em.color[0]),
                    1.0f - clamp01(em.color[1]),
                    1.0f - clamp01(em.color[2])
                );

                #pragma omp parallel for schedule(static)
                for (int i = 0; i < NX*NY; ++i) {
                    if (!em.mask[i]) continue;

                    float inj = EMISOR_RATE_MASA;

                    masa.a[i] += inj;
                    absorcion_premul.a[i] += cv::Vec3f(inj*abs_color[0], inj*abs_color[1], inj*abs_color[2]);

                    u.a[i] += em.vx;
                    v.a[i] += em.vy;
                }
                vivos.push_back(em);
            }
        }
        emisores.swap(vivos);

        // clamp + decay
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < NX*NY; ++i) {
            masa.a[i] = std::min(MASA_MAX, std::max(0.0f, masa.a[i]));
            masa.a[i] *= (1.0f - DECAY_MASA);
            absorcion_premul.a[i] *= (1.0f - DECAY_MASA);
        }

        // difundir velocidad
        u = difundir_jacobi_mascara(u, VISC, DT, 8, m);
        v = difundir_jacobi_mascara(v, VISC, DT, 8, m);
        aplicar_bordes_vel(u, v, m);

        // vorticidad
        confinamiento_vorticidad(u, v, VORT_EPS, m);

        // gravedad/decantación (solo si densidad > 1)
        float offset_dens = std::max(0.0f, DENSIDAD_PINTURA - 1.0f);
        if (offset_dens > 0.0f && GRAVEDAD != 0.0f) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < NX*NY; ++i) v.a[i] += (GRAVEDAD * offset_dens) * masa.a[i] * m.a[i];
            aplicar_bordes_vel(u, v, m);
        }

        // proyectar
        proyectar(u, v, JACOBI_ITERS, m);
        aplicar_bordes_vel(u, v, m);

        // advectar velocidad
        u = advectar_escalar(u, u, v, DT, m);
        v = advectar_escalar(v, u, v, DT, m);
        aplicar_bordes_vel(u, v, m);

        // proyectar de nuevo
        proyectar(u, v, JACOBI_ITERS, m);
        aplicar_bordes_vel(u, v, m);

        // difundir masa
        masa = difundir_jacobi_mascara(masa, DIFF_MASA, DT, 6, m);

        // difundir absorción (color)
        Campo ar(0), ag(0), ab(0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < NX*NY; ++i) {
            ar.a[i] = absorcion_premul.a[i][0];
            ag.a[i] = absorcion_premul.a[i][1];
            ab.a[i] = absorcion_premul.a[i][2];
        }
        if (DIFF_COLOR > 0.0f) {
            ar = difundir_jacobi_mascara(ar, DIFF_COLOR, DT, 6, m);
            ag = difundir_jacobi_mascara(ag, DIFF_COLOR, DT, 6, m);
            ab = difundir_jacobi_mascara(ab, DIFF_COLOR, DT, 6, m);
        }

        // advectar masa y absorción
        masa = advectar_escalar(masa, u, v, DT, m);
        Campo ar2 = advectar_escalar(ar, u, v, DT, m);
        Campo ag2 = advectar_escalar(ag, u, v, DT, m);
        Campo ab2 = advectar_escalar(ab, u, v, DT, m);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < NX*NY; ++i) {
            masa.a[i] = std::min(MASA_MAX, std::max(0.0f, masa.a[i])) * m.a[i];
            absorcion_premul.a[i] = cv::Vec3f(ar2.a[i], ag2.a[i], ab2.a[i]) * m.a[i];
        }

        // render
        small.setTo(cv::Scalar(0,0,0));
        const float k_opacidad = 0.55f;

        #pragma omp parallel for schedule(static)
        for (int y = 0; y < NY; ++y) {
            cv::Vec3b* row = small.ptr<cv::Vec3b>(y);
            for (int x = 0; x < NX; ++x) {
                if (m.at(x,y) == 0.0f) {
                    cv::Vec3f f = fondo_rgb;
                    row[x] = cv::Vec3b(
                        (uchar)std::lround(255.0f * clamp01(f[2])),
                        (uchar)std::lround(255.0f * clamp01(f[1])),
                        (uchar)std::lround(255.0f * clamp01(f[0]))
                    );
                    continue;
                }

                float mm = masa.at(x,y);
                cv::Vec3f ap = absorcion_premul.at(x,y);

                float denom = mm + 1e-6f;
                cv::Vec3f abs_media(ap[0]/denom, ap[1]/denom, ap[2]/denom);

                abs_media[0] = clamp01(abs_media[0]);
                abs_media[1] = clamp01(abs_media[1]);
                abs_media[2] = clamp01(abs_media[2]);

                cv::Vec3f color_tinte(
                    1.0f - abs_media[0],
                    1.0f - abs_media[1],
                    1.0f - abs_media[2]
                );

                color_tinte[0] = clamp01(color_tinte[0]);
                color_tinte[1] = clamp01(color_tinte[1]);
                color_tinte[2] = clamp01(color_tinte[2]);

                float alpha = 1.0f - std::exp(-k_opacidad * mm);
                alpha = clamp01(alpha);

                cv::Vec3f final_rgb = fondo_rgb * (1.0f - alpha) + color_tinte * alpha;

                row[x] = cv::Vec3b(
                    (uchar)std::lround(255.0f * clamp01(final_rgb[2])),
                    (uchar)std::lround(255.0f * clamp01(final_rgb[1])),
                    (uchar)std::lround(255.0f * clamp01(final_rgb[0]))
                );
            }
        }

        cv::resize(small, shown, cv::Size(W,H), 0, 0, cv::INTER_NEAREST);

        // línea de agua (si la quieres quitar, elimina estas 2 líneas)
        cv::line(shown, cv::Point(0, WATERLINE_Y * CELL), cv::Point(W, WATERLINE_Y * CELL),
                 cv::Scalar(255,255,255), 1);

        if (GUARDAR_VIDEO && writer.isOpened()) {
            writer.write(shown);
        }

        int key = -1;
        if ((frame_en_segmento % MOSTRAR_CADA_N_FRAMES) == 0) {
            cv::imshow("Fluido", shown);
            key = cv::waitKey(1) & 0xFF;
        }

        frame_en_segmento++;

        if ((frame_en_segmento % IMPRIMIR_CADA_FRAMES) == 0) {
            imprimir_stats_segmento(frame_en_segmento, inicio_segmento, nombre_video);
        }

        if (key == 27) break;
    }

    if (GUARDAR_VIDEO && writer.isOpened()) {
        writer.release();
        std::cout << "[video] cerrado: " << nombre_video << "\n" << std::flush;
    }
    cv::destroyAllWindows();
    return 0;
}

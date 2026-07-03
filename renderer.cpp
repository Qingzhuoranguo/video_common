#include "renderer.h"

#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLContext>

namespace {

static const float kQuadVertices[] = {
    // x      y     u     v
    -1.0f,  1.0f, 0.0f, 0.0f,
    -1.0f, -1.0f, 0.0f, 1.0f,
    1.0f,  1.0f, 1.0f, 0.0f,
    1.0f, -1.0f, 1.0f, 1.0f,
};

static const char* kVertSrc = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
uniform vec2 uQuadScale;
out vec2 vUV;
void main() {
    gl_Position = vec4(aPos * uQuadScale, 0.0, 1.0);
    vUV = aUV;
}
)";

static const char* kFragSrc = R"(
#version 330 core
in vec2 vUV;
uniform sampler2D uTex;
out vec4 fragColor;
void main() {
    fragColor = texture(uTex, vUV);
}
)";

} // namespace

struct Renderer::Impl_ {
    uint32_t    textureId   = 0;
    uint32_t    texWidth    = 0;
    uint32_t    texHeight   = 0;
    DisplayMode displayMode = DisplayMode::Fit;
    int         viewportW   = 0;
    int         viewportH   = 0;

    void* nativeGLContext = nullptr;

    QOpenGLShaderProgram*    program = nullptr;
    QOpenGLBuffer            vbo{QOpenGLBuffer::VertexBuffer};
    QOpenGLVertexArrayObject vao;

    int locQuadScale = -1;
    int locTex       = -1;

    void quadScale(float& sx, float& sy) const {
        sx = 1.0f; sy = 1.0f;
        if (displayMode != DisplayMode::Fit) return;
        if (texWidth == 0 || texHeight == 0 || viewportW == 0 || viewportH == 0) return;

        const float texAR  = static_cast<float>(texWidth)  / static_cast<float>(texHeight);
        const float viewAR = static_cast<float>(viewportW) / static_cast<float>(viewportH);

        if (texAR > viewAR) {
            sy = viewAR / texAR;
        } else {
            sx = texAR / viewAR;
        }
    }
};

Renderer::Renderer(QWidget* parent)
    : QOpenGLWidget(parent), m_impl(new Impl_()) {}

Renderer::~Renderer() {
    makeCurrent();
    delete m_impl->program;
    m_impl->vbo.destroy();
    m_impl->vao.destroy();
    doneCurrent();
    delete m_impl;
}

void* Renderer::glContext() const {
    return m_impl->nativeGLContext;
}

void Renderer::setTexture(uint32_t textureId, uint32_t width, uint32_t height) {
    m_impl->textureId = textureId;
    m_impl->texWidth  = width;
    m_impl->texHeight = height;
    update();
}

void Renderer::setDisplayMode(DisplayMode mode) {
    m_impl->displayMode = mode;
    update();
}

void Renderer::initializeGL() {
    initializeOpenGLFunctions();
    glClearColor(0.12f, 0.12f, 0.12f, 1.0f);

    m_impl->program = new QOpenGLShaderProgram(this);
    m_impl->program->addShaderFromSourceCode(QOpenGLShader::Vertex,   kVertSrc);
    m_impl->program->addShaderFromSourceCode(QOpenGLShader::Fragment, kFragSrc);
    m_impl->program->bindAttributeLocation("aPos", 0);
    m_impl->program->bindAttributeLocation("aUV",  1);
    m_impl->program->link();

    m_impl->locQuadScale = m_impl->program->uniformLocation("uQuadScale");
    m_impl->locTex       = m_impl->program->uniformLocation("uTex");

    m_impl->vao.create();
    m_impl->vao.bind();

    m_impl->vbo.create();
    m_impl->vbo.setUsagePattern(QOpenGLBuffer::StaticDraw);
    m_impl->vbo.bind();
    m_impl->vbo.allocate(kQuadVertices, sizeof(kQuadVertices));

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          reinterpret_cast<void*>(2 * sizeof(float)));

    m_impl->vao.release();
    m_impl->vbo.release();

    // 提取底层 GLXContext，QOpenGLContext 头文件已包含 QNativeInterface
    auto* native = QOpenGLContext::currentContext()
                       ->nativeInterface<QNativeInterface::QGLXContext>();
    if (native) {
        m_impl->nativeGLContext = reinterpret_cast<void*>(native->nativeContext());
    }
}

void Renderer::resizeGL(int w, int h) {
    m_impl->viewportW = w;
    m_impl->viewportH = h;
}

void Renderer::paintGL() {
    glClear(GL_COLOR_BUFFER_BIT);
    if (m_impl->textureId == 0) return;

    float sx, sy;
    m_impl->quadScale(sx, sy);

    m_impl->program->bind();
    m_impl->vao.bind();

    m_impl->program->setUniformValue(m_impl->locQuadScale, sx, sy);
    m_impl->program->setUniformValue(m_impl->locTex, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(m_impl->textureId));

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glBindTexture(GL_TEXTURE_2D, 0);
    m_impl->vao.release();
    m_impl->program->release();
}

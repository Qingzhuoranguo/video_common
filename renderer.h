#pragma once

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <cstdint>

enum class DisplayMode {
    Fill,
    Stretch,
    Fit,
};
class Renderer : public QOpenGLWidget, protected QOpenGLFunctions {
    Q_OBJECT

public:

    explicit Renderer(QWidget* parent = nullptr);
    ~Renderer() override;

    void* glContext() const;

    void setTexture(uint32_t textureId, uint32_t width, uint32_t height);
    void setDisplayMode(DisplayMode mode);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

private:
    struct Impl_;
    Impl_* m_impl;
};

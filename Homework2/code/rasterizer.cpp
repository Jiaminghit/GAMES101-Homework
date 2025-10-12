// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#define MSAA true

rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(int x, int y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    Eigen::Vector3f p0p1(_v[1].x() - _v[0].x(), _v[1].y() - _v[0].y(), 1.0f);
    Eigen::Vector3f p1p2(_v[2].x() - _v[1].x(), _v[2].y() - _v[1].y(), 1.0f);
    Eigen::Vector3f p2p0(_v[0].x() - _v[2].x(), _v[0].y() - _v[2].y(), 1.0f);

    Eigen::Vector3f p0p(x - _v[0].x(), y - _v[0].y(), 1.0f);
    Eigen::Vector3f p1p(x - _v[1].x(), y - _v[1].y(), 1.0f);
    Eigen::Vector3f p2p(x - _v[2].x(), y - _v[2].y(), 1.0f);

    if (p0p1.cross(p0p).z() > 0.f) {
        return p1p2.cross(p1p).z() > 0.f && p2p0.cross(p2p).z() > 0.f;
    } else {
        return p1p2.cross(p1p).z() < 0.f && p2p0.cross(p2p).z() < 0.f;
    }
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }

    if (MSAA) {
        for (int x = 0; x < width; x++){
            for (int y = 0; y < height; y++){
                Eigen::Vector3f color(0, 0, 0);
                for (int i = 0; i < 4; i++) 
                    color += frame_buf_2XMSAA[get_index(x, y) + i];
                color /= 4;
                set_pixel(Eigen::Vector3f(x, y, 1.0f), color);
            }
        }
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    
    // TODO : Find out the bounding box of current triangle.
    std::vector<float> x_arry{v[0].x(), v[1].x(), v[2].x()};
    std::vector<float> y_arry{v[0].y(), v[1].y(), v[2].y()};
    std::sort(x_arry.begin(), x_arry.end());
    std::sort(y_arry.begin(), y_arry.end());
    int x_min = floor(x_arry[0]), x_max = ceil(x_arry[2]),
        y_min = floor(y_arry[0]), y_max = ceil(y_arry[2]);
    // iterate through the pixel and find if the current pixel is inside the triangle
    // 光栅化
    for (int x = x_min; x < x_max; x++) {
        for (int y = y_min; y < y_max; y++) {
            if (x < 0 || x > width || y < 0 || y > height) {
                continue;
            }
            if (MSAA){
                int index = 0;
                for (float i = 0.25; i < 1.0; i += 0.5) {
                    for (float j = 0.25; j < 1.0; j += 0.5) {
                        if (insideTriangle(x + i, y + j, t.v)) {
                            auto[alpha, beta, gamma] = computeBarycentric2D(x + i, y + j, t.v);
                            float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                            float z_interpolated = 
                                alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                            z_interpolated *= w_reciprocal;
                            int index_ans = get_index(x, y) + index;
                            if (abs(z_interpolated) < depth_buf_2XMSAA[index_ans]) {
                            frame_buf_2XMSAA[index_ans] = t.getColor();
                            depth_buf_2XMSAA[index_ans] = abs(z_interpolated);
                            }
                        }
                    }
                }
            } else {
                if (insideTriangle(x + 0.5f, y + 0.5f, t.v)) {
                    // If so, use the following code to get the interpolated z value.
                    auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
                    float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = 
                        alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;
// TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
                    if (abs(z_interpolated) < depth_buf[get_index(x, y)]) {
                        Eigen::Vector3f point(x, y, 1.0f);
                        set_pixel(point, t.getColor());
                        depth_buf[get_index(x, y)] = abs(z_interpolated);
                    }
                }
            }
        }
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
        // if MSAA
        std::fill(frame_buf_2XMSAA.begin(), frame_buf_2XMSAA.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
        std::fill(depth_buf_2XMSAA.begin(), depth_buf_2XMSAA.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    // if MSAA
    frame_buf_2XMSAA.resize(w * h * 4);
    depth_buf_2XMSAA.resize(w * h * 4);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on
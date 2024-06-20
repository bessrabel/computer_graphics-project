#include <iostream>
#include <vector>
#include <stdio.h>
#include <algorithm>

#include <glad.h>
#include <glfw3.h>
#include <glm.hpp>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
using namespace cv;
using namespace glm;
using namespace Eigen;

struct MyPoint {
    int x, y;

    bool operator==(const MyPoint& other) const {
        return x == other.x && y == other.y;
    }
};

// 로컬 적응형 임계값 방식으로 이미지 세분화
unsigned char* adaptiveThreshold(unsigned char* img_data, int width, int height, int channels) {
    
    unsigned char* binaryImage = new unsigned char[width * height];

    int blockSize = 7;  // 각 픽셀에 대한 윈도우 크기
    int C = 0;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sum = 0;
            int count = 0;
            for (int dy = -blockSize / 2; dy <= blockSize / 2; dy++) {
                for (int dx = -blockSize / 2; dx <= blockSize / 2; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int index = (ny * width + nx) * channels; 
                        int pixelValue = 0.2989 * img_data[index] + 0.5870 * img_data[index + 1] + 0.1140 * img_data[index + 2];  // 회색 이미지 처리 사용 
                        sum += pixelValue;
                        count++;
                    }
                }
            }
            int index = y * width + x;
            int mean = sum / count;
            binaryImage[index] = (img_data[index * channels] < mean -C) ? 255 : 0;  // 픽셀과 평균 비교
        }
    }
    return binaryImage;
}

// 윤곽선 검출 함수 (Suzuki와 Abe 알고리즘)
void findContours(const unsigned char* binaryImage, int width, int height, std::vector<std::vector<MyPoint>>& contours) {
    vector<vector<bool>> visited(height, vector<bool>(width, false));
    const MyPoint directions[8] = {{1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}};

    auto isInside = [width, height](int x, int y) {
        return x >= 0 && x < width && y >= 0 && y < height;
    };

    auto nextDirection = [](int dir) {
        return (dir + 1) % 8;
    };

    auto previousDirection = [](int dir) {
        return (dir + 7) % 8;
    };

    auto findNextPoint = [&](const MyPoint& p, int dir, int& newDir) {
        for (int i = 0; i < 8; ++i) {
            int tempDir = (dir + i) % 8;
            int nx = p.x + directions[tempDir].x;
            int ny = p.y + directions[tempDir].y;
            if (isInside(nx, ny) && binaryImage[ny * width + nx] == 255 && !visited[ny][nx]) {
                newDir = tempDir;
                return MyPoint{nx, ny};
            }
        }
        newDir = -1;
        return MyPoint{-1, -1};
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (binaryImage[y * width + x] == 255 && !visited[y][x]) {
                vector<MyPoint> contour;
                MyPoint start = {x, y};
                MyPoint current = start;
                int dir = 0;

                do {
                    contour.push_back(current);
                    visited[current.y][current.x] = true;
                    int newDir;
                    MyPoint next = findNextPoint(current, (dir + 7) % 8, newDir);
                    if (newDir == -1) break;  // 더 이상 갈 곳이 없으면 종료
                    current = next;
                    dir = newDir;
                } while (current.x != start.x || current.y != start.y);

                if (contour.size() > 500 && contour.size() < 1200) {
                    contours.push_back(contour);
                }
            }
        }
    }
}

void douglasPeucker(const vector<MyPoint>& points, double epsilon, vector<MyPoint>& out) {
    if (points.size() < 2) {
        out = points;
        return;
    }

    double maxDist = 0;
    int index = 0;

    MyPoint start = points.front();
    MyPoint end = points.back();

    for (int i = 1; i < points.size() - 1; ++i) {
        double dist = abs((end.y - start.y) * points[i].x - (end.x - start.x) * points[i].y + end.x * start.y - end.y * start.x) / 
                      sqrt(pow(start.x - end.x, 2) + pow(start.y - end.y, 2));
        if (dist > maxDist) {
            maxDist = dist;
            index = i;
        }
    }

    if (maxDist > epsilon) {
        vector<MyPoint> rec1, rec2;
        vector<MyPoint> first(points.begin(), points.begin() + index + 1);
        vector<MyPoint> second(points.begin() + index, points.end());
        douglasPeucker(first, epsilon, rec1);
        douglasPeucker(second, epsilon, rec2);
        out.assign(rec1.begin(), rec1.end() - 1);
        out.insert(out.end(), rec2.begin(), rec2.end());
    } else {
        out = {start, end};
    }
}

void convertToQuadrilateral(vector<MyPoint>& contour, int width, int height) {
    if (contour.size() >= 4) {
        vector<MyPoint> bestQuad;
        double bestArea = 0;
        double aspectRatioThreshold = 5;  // 가로와 세로의 비율이 이 임계값 내에 있어야 함

        for (size_t i = 0; i < contour.size(); ++i) {
            for (size_t j = i + 1; j < contour.size(); ++j) {
                for (size_t k = j + 1; k < contour.size(); ++k) {
                    for (size_t l = k + 1; l < contour.size(); ++l) {
                        vector<MyPoint> quad = {contour[i], contour[j], contour[k], contour[l]};

                        // 중심점을 기준으로 점을 시계 방향으로 정렬
                        auto centroid = MyPoint{(quad[0].x + quad[1].x + quad[2].x + quad[3].x) / 4,
                                              (quad[0].y + quad[1].y + quad[2].y + quad[3].y) / 4};
                        sort(quad.begin(), quad.end(), [centroid](const MyPoint& a, const MyPoint& b) {
                            double angleA = atan2(a.y - centroid.y, a.x - centroid.x);
                            double angleB = atan2(b.y - centroid.y, b.x - centroid.x);
                            return angleA < angleB;
                        });

                        // 사각형의 가로와 세로 길이 계산
                        double width1 = sqrt(pow(quad[1].x - quad[0].x, 2) + pow(quad[1].y - quad[0].y, 2));
                        double width2 = sqrt(pow(quad[3].x - quad[2].x, 2) + pow(quad[3].y - quad[2].y, 2));
                        double height1 = sqrt(pow(quad[2].x - quad[1].x, 2) + pow(quad[2].y - quad[1].y, 2));
                        double height2 = sqrt(pow(quad[0].x - quad[3].x, 2) + pow(quad[0].y - quad[3].y, 2));
                        double avgWidth = (width1 + width2) / 2.0;
                        double avgHeight = (height1 + height2) / 2.0;

                        double aspectRatio = abs(avgWidth - avgHeight) / std::min(avgWidth, avgHeight);
                        double area = avgWidth * avgHeight;

                        // 가로와 세로의 비율이 임계값 내에 있는 후보 중에서 가장 큰 사각형 선택
                        if (aspectRatio <= aspectRatioThreshold && area > bestArea) {
                            bestArea = area;
                            bestQuad = quad;
                        }
                    }
                }
            }
        }

        if (!bestQuad.empty()) {
            // 좌표가 이미지 경계를 넘어가지 않도록 조정
            for (auto& pt : bestQuad) {
                pt.x = std::max(0, std::min(pt.x, width - 1));
                pt.y = std::max(0, std::min(pt.y, height - 1));
            }

            contour = bestQuad;
        }
    }
}

// 외부 윤곽만 남김
void removeInternalContours(vector<vector<MyPoint>>& contours) {
    if (contours.empty()) return;

    // bounding box 계산
    auto boundingBox = [](const vector<MyPoint>& contour) {
        struct BoundingBox {
            int minX, minY, maxX, maxY;
        };
        BoundingBox box;
        box.minX = contour[0].x;
        box.minY = contour[0].y;
        box.maxX = contour[0].x;
        box.maxY = contour[0].y;
        for (const auto& pt : contour) {
            if (pt.x < box.minX) box.minX = pt.x;
            if (pt.y < box.minY) box.minY = pt.y;
            if (pt.x > box.maxX) box.maxX = pt.x;
            if (pt.y > box.maxY) box.maxY = pt.y;
        }
        return box;
    };

    vector<typename decltype(boundingBox(contours[0]))::BoundingBox> boundingBoxes;
    for (const auto& contour : contours) {
        boundingBoxes.push_back(boundingBox(contour));
    }

    vector<bool> keep(contours.size(), true);

    for (size_t i = 0; i < contours.size(); ++i) {
        for (size_t j = 0; j < contours.size(); ++j) {
            if (i != j) {
                if (boundingBoxes[i].minX <= boundingBoxes[j].minX && boundingBoxes[i].minY <= boundingBoxes[j].minY &&
                    boundingBoxes[i].maxX >= boundingBoxes[j].maxX && boundingBoxes[i].maxY >= boundingBoxes[j].maxY) {
                    keep[j] = false;
                }
            }
        }
    }

    vector<vector<MyPoint>> filteredContours;
    for (size_t i = 0; i < contours.size(); ++i) {
        if (keep[i]) {
            filteredContours.push_back(contours[i]);
        }
    }
    contours = filteredContours;
}

// 투시 변환을 수동으로 계산하고 적용하는 함수
Matrix3f computeHomography(const vector<MyPoint>& src, const vector<MyPoint>& dst) {
    MatrixXf A(8, 9);
    A.setZero();

    for (int i = 0; i < 4; ++i) {
        int j = 2 * i;
        A(j, 0) = src[i].x;
        A(j, 1) = src[i].y;
        A(j, 2) = 1;
        A(j, 6) = -dst[i].x * src[i].x;
        A(j, 7) = -dst[i].x * src[i].y;
        A(j, 8) = -dst[i].x;

        A(j + 1, 3) = src[i].x;
        A(j + 1, 4) = src[i].y;
        A(j + 1, 5) = 1;
        A(j + 1, 6) = -dst[i].y * src[i].x;
        A(j + 1, 7) = -dst[i].y * src[i].y;
        A(j + 1, 8) = -dst[i].y;
    }

    JacobiSVD<MatrixXf> svd(A, ComputeFullV);
    VectorXf h = svd.matrixV().col(8);
    Matrix3f H;
    H << h(0), h(1), h(2),
         h(3), h(4), h(5),
         h(6), h(7), h(8);

    return H;
}

// 투시변환 적용
MyPoint applyHomography(const Matrix3f& H, const MyPoint& pt) {
    Vector3f src(pt.x, pt.y, 1.0f);
    Vector3f dst = H * src;
    dst /= dst.z();

    return {static_cast<int>(round(dst.x())), static_cast<int>(round(dst.y()))};
}

// 마커 코드를 추출하는 함수
vector<vector<int>> extractMarkerCode(const vector<MyPoint>& contour, const unsigned char* img_data, int width, int height, int channels) {
    vector<MyPoint> srcPoints = contour;
    vector<MyPoint> dstPoints = {
        {0, 0}, {48, 0}, {48, 48}, {0, 48}
    };

    Matrix3f H = computeHomography(srcPoints, dstPoints);
    vector<unsigned char> warped(49 * 49 * channels, 0);
    Matrix3f H_inv = H.inverse();
    for (int y = 0; y < 49; ++y) {
        for (int x = 0; x < 49; ++x) {
            MyPoint pt = applyHomography(H_inv, { x, y });
            int srcX = std::min(std::max((int)round(pt.x), 0), width - 1);
            int srcY = std::min(std::max((int)round(pt.y), 0), height - 1);
            if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
                for (int c = 0; c < channels; ++c) {
                    int a = img_data[(srcY * width + srcX) * channels + c];
                    warped[(y * 49 + x) * channels + c] = a;
                }
            } else {
                // 변환된 좌표가 유효하지 않은 경우를 처리합니다.
                for (int c = 0; c < channels; ++c) {
                    warped[(y * 49 + x) * channels + c] = 255; // 흰색으로 채워서 문제를 시각적으로 표시합니다.
                }
            }
        }
    }

    int gridSize = 7;
    vector<vector<int>> markerCode(gridSize, vector<int>(gridSize, 0));
    int cellSize = 7;

    for (int y = 0; y < gridSize; y++) {
        for (int x = 0; x < cellSize; x++) {
            int count = 0;
            for (int dy = 0; dy < gridSize; ++dy) {
                for (int dx = 0; dx < cellSize; ++dx) {
                    int index = ((y * cellSize + dy) * 49 + (x * cellSize + dx)) * channels;
                    int value = 0.2989 * warped[index] + 0.5870 * warped[index + 1] + 0.1140 * warped[index + 2];
                    if (value > 110) {
                        count++;
                    }

                }
            }
            int th = gridSize*cellSize/2;
            markerCode[y][x] = (count > th ) ? 1 : 0;
        }
    }

    return markerCode;
}

// 마커의 검은색 경계를 확인하는 함수
bool isBlackBorder(const vector<vector<int>>& markerCode) {
    int gridSize = markerCode.size();
    for (int i = 0; i < gridSize; ++i) {
        if (markerCode[0][i] != 0 || markerCode[gridSize - 1][i] != 0 ||
            markerCode[i][0] != 0 || markerCode[i][gridSize - 1] != 0) {
            return false;
        }
    }
    return true;
}

unsigned char* drawContoursOnImage(unsigned char* img_data, int width, int height, int channels, const vector<vector<MyPoint>>& contours, int r, int g, int b) {
    unsigned char* img_copy = new unsigned char[width * height * channels];
    memcpy(img_copy, img_data, width * height * channels);

    for (const auto& contour : contours) {
        for (size_t i = 0; i < contour.size(); ++i) {
            int x1 = contour[i].x;
            int y1 = contour[i].y;
            int x2 = contour[(i + 1) % contour.size()].x;
            int y2 = contour[(i + 1) % contour.size()].y;

            // Bresenham's line algorithm to draw line between points
            int dx = abs(x2 - x1), sx = x1 < x2 ? 1 : -1;
            int dy = abs(y2 - y1), sy = y1 < y2 ? 1 : -1;
            int err = (dx > dy ? dx : -dy) / 2, e2;

            while (true) {
                if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
                    int index = (y1 * width + x1) * channels;
                    img_copy[index] = r;
                    img_copy[index + 1] = g;
                    img_copy[index + 2] = b;
                    if (channels == 4) {
                        img_copy[index + 3] = 255;
                    }
                }
                if (x1 == x2 && y1 == y2) break;
                e2 = err;
                if (e2 > -dx) { err -= dy; x1 += sx; }
                if (e2 < dy) { err += dx; y1 += sy; }
            }
        }
    }
    return img_copy;
}

Mat saveMarkerImageWithHomographySimple(const vector<MyPoint>& contour, const unsigned char* img_data, int width, int height, int channels) {
    vector<MyPoint> srcPoints = contour;
    vector<MyPoint> dstPoints = {
        {0, 0}, {48, 0}, {48, 48}, {0, 48}
    };

    Matrix3f H = computeHomography(srcPoints, dstPoints);
    Matrix3f H_inv = H.inverse();

    // 원래 이미지 복사본 생성
    //Mat img_copy(height, width, (channels == 3 ? CV_8UC3 : CV_8UC4), (void*)img_data);
    Mat img_copys(height, width, (channels == 3 ? CV_8UC3 : CV_8UC4), (void*)img_data);
    Mat img_copy = img_copys.clone();

    // 3D 축 정의 (x, y, z 축)
    vector<Vector3f> axis = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}
    };

    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R(i, j) = H_inv(i, j);
        }
        t(i) = H_inv(i, 2);
    }

    // 카메라 매트릭스 설정
    Mat cameraMatrix = (Mat_<double>(3, 3) << width + height / 2, 0, width / 2, 0, width + height / 2, height / 2, 0, 0, 1);
    Mat distCoeffs = Mat::zeros(5, 1, CV_64F);

    // 마커 중심 계산
    MyPoint center = {(srcPoints[0].x + srcPoints[1].x + srcPoints[2].x + srcPoints[3].x) / 4,
                      (srcPoints[0].y + srcPoints[1].y + srcPoints[2].y + srcPoints[3].y) / 4};

    vector<Point> imagePoints(4);
    // 2D 이미지 좌표로 변환
    for (size_t i = 0; i < axis.size(); ++i) {
        Vector3f pt = axis[i];
        Eigen::Vector3f point2D = R * pt;
        point2D /= point2D.z();
        if (i == 3) { // Z축만 변환
            float fx = cameraMatrix.at<double>(0, 0);
            float fy = cameraMatrix.at<double>(1, 1);
            float cx = cameraMatrix.at<double>(0, 2);
            float cy = cameraMatrix.at<double>(1, 2);

            float x = point2D.x() * fx + cx;
            float y = point2D.y() * fy + cy;
            imagePoints[i] = Point(x, y);
        } else if(i ==2 || i == 1) { // X축과 Y축은 단순 투영
            float x = point2D.x();
            float y = point2D.y();
            imagePoints[i] = Point(x,y);
        }
    }

    // 축 그리기
    line(img_copy, Point2f(center.x, center.y), imagePoints[1], Scalar(0, 0, 255), 2); // X축 (빨강)
    line(img_copy, Point2f(center.x, center.y), imagePoints[2], Scalar(0, 255, 0), 2); // Y축 (초록)
    line(img_copy, Point2f(center.x, center.y), imagePoints[3], Scalar(255, 0, 0), 2); // Z축 (파랑)

    // 고유한 파일 이름 생성
    static int imageCounter = 0;
    char filename[50];
    snprintf(filename, sizeof(filename), "homography_image_%d.jpg", imageCounter);
    imageCounter++;

    imwrite(filename, img_copy);
    return img_copy;
}


int main() {
    const char* filename = "input.PNG"; // 이미지 파일 이름
    int width, height, channels;
    unsigned char* image_data = stbi_load(filename, &width, &height, &channels, 0);
    if (!image_data) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return -1;
    }
    unsigned char* binaryImage = adaptiveThreshold(image_data, width, height, channels);

    stbi_write_jpg("grayscale_image.jpg", width, height, 1, binaryImage, 90);

    // 윤곽선 찾기
    std::vector<std::vector<MyPoint>> contours;
    findContours(binaryImage, width, height, contours);

    // 다각형 근사
    double epsilon = 2; // 근사 임계값
    std::vector<std::vector<MyPoint>> approxContours;
    for (const auto& contour : contours) {
        std::vector<MyPoint> approx;
        douglasPeucker(contour, epsilon, approx);
        
        if (approx.size() >= 2 && approx.size() < 20) {
            convertToQuadrilateral(approx, width, height);
            approxContours.push_back(approx);
        }
    }

    removeInternalContours(approxContours);
    unsigned char* greenImage = drawContoursOnImage(image_data, width, height, channels, approxContours, 0, 255, 0);

    // 윤곽선을 원래 이미지에 그리기
    unsigned char* redImage = drawContoursOnImage(image_data, width, height, channels, contours, 255, 0, 0);

    stbi_write_jpg("red_image.jpg", width, height, channels, redImage, width * channels);
    stbi_write_jpg("green_image.jpg", width, height, channels, greenImage, width * channels);

    for (const auto& contour : approxContours) {
        std::vector<std::vector<int>> markerCode = extractMarkerCode(contour, image_data, width, height, channels);
        if (isBlackBorder(markerCode)) {
            Mat rvec, tvec;
            Mat imagecopy = saveMarkerImageWithHomographySimple(contour, image_data, width, height, channels);
            imshow("202372001 김소영 Detected Markers", imagecopy);
            waitKey(0);
        }
    }
    return 0;
}
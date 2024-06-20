# computer_graphics final porject

## Maker Detection
---

## 코드 변경사항
기존 bunny.obj를 띄우는 대신 포즈 추정 결과를 확인하기 위해 openCV를 추가로 설치해 마커 위에 x,y,z 축을 그리는 것으로 대신함.

### Prerequisites
#### 1. Visual Studio Code & Extensions
Visual Studio Code 설치 및 아래의 2개 확장 설치 필요

+ CMake

![CMake](https://github.com/bessrabel/computer_graphics-hw4/blob/main/readmePng/cmake.PNG)

+ CMake Tools

![CMake Tools](https://github.com/bessrabel/computer_graphics-hw4/blob/main/readmePng/cmakeTools.PNG)

#### 2. g++
Using GCC with MinGW64

설치 참조: <https://code.visualstudio.com/docs/cpp/config-mingw#_create-hello-world> **PATH 추가 필수**

#### 3. CMake 
Using CMake

설치 참조: <https://cmake.org/download/> **version 3.29.2**

#### 4. openCV

설치 참조: <https://github.com/huihut/OpenCV-MinGW-Build?tab=readme-ov-file> **version 4.5.2 -x64**

---

### Directory Layout
> + include
> + lib
> + src
> + readmePng
> + screenshot


#### include & lib
라이브러리 폴더
+ GLFW (Version 3.3.bin.win32)
+ GLAD (gl: 4.6, profile: Core, extensions: none)
+ GLM (Version 0.9.9.6)
+ openCV(version 4.5.2)
+ lib (필요한 여러 dil 파일)

#### readmePng
README.md 파일 이미지 첨부를 위한 폴더

#### resultImg
코드 실행 결과 저장 이미지 파일

#### src
소스 코드 폴더 (.cpp파일)
+ input.PNG : 마커가 출력된 이미지 파일
+ main.cpp : 마커 검출 하는 main 파일  
+ stb_image_write.h & tb_image.h : 이미지 입출력 관련 함수 헤더 파일
---

### compilation instructions

```
1. vscode를 실행 후, 다운받은 파일 폴더를 프로젝트 폴더로 선택 
2. 명령창(F1 단축키)으로 CMake:configure 명령 선택하여 운영체제에 맞는 컴파일러 도구(gcc 등) 선택
3. 다시 command를 입력할 수 있는 명령창을 열고 CMake:Build로 빌드(이때 CMakeList.txt 파일을 참고하여 자동으로 빌드됨)
4. 마지막으로 디버그(명령창 CMake:dubug or ctrl+ F5)하여 실행 결과를 확인
```

+ cmakeList.txt 파일 실행 시 *find_package( OpenCV REQUIRED )* 가 자동으로 검색이 안되어 수동으로  *set(OpenCV_DIR C:/opencv-4.5.2-mingw/)* 입력함.
+ 따라서 opencv 경로가 다르면 CmakeList.txt 내용을 다르게 설정해야 함.

 ---
 
### result

![input_a](https://github.com/bessrabel/computer_graphics-project/blob/main/resultImg/homography_image_0.jpg)


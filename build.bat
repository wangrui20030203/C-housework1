@echo off
rem ============================================================
rem  build.bat - 编译自动检测计数程序
rem  在 Visual Studio 的 "x64 Native Tools Command Prompt" 中运行
rem  或直接双击本文件
rem ============================================================
setlocal

set "VS_BUILD=D:\VS_Cjj\Microsoft\product"
set "OPENCV_BUILD=D:\opencv\opencv_4.60\opencv\build"

call "%VS_BUILD%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] vcvars64.bat 调用失败，请检查 VS_BUILD 变量
    exit /b 1
)

rem ---- 部分机器的 Windows SDK 不在默认盘，vcvars 无法自动解析，此处兜底补齐 ----
if "%WindowsSdkDir%"=="" (
    if exist "D:\Windows Kits\10\Include\10.0.26100.0\ucrt" (
        set "INCLUDE=D:\Windows Kits\10\Include\10.0.26100.0\ucrt;D:\Windows Kits\10\Include\10.0.26100.0\shared;D:\Windows Kits\10\Include\10.0.26100.0\um;D:\Windows Kits\10\Include\10.0.26100.0\winrt;%INCLUDE%"
        set "LIB=D:\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64;D:\Windows Kits\10\Lib\10.0.26100.0\um\x64;%LIB%"
        echo [INFO] 已手动补齐 Windows SDK 路径
    ) else (
        echo [WARN] 未能解析 Windows SDK，若编译报 float.h 缺失请手动设置 INCLUDE/LIB
    )
)

cl /EHsc /nologo /std:c++17 /utf-8 /O2 /Fe:obj_count.exe main.cpp ^
   /I"%OPENCV_BUILD%\include" ^
   /link /LIBPATH:"%OPENCV_BUILD%\x64\vc15\lib" opencv_world460.lib
if errorlevel 1 (
    echo.
    echo [ERROR] 编译失败
    exit /b 1
)

del /q *.obj 2>nul
echo.
echo [OK] 生成 obj_count.exe
exit /b 0

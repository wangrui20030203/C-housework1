/**
 * ============================================================
 *  物体自动检测与计数 (Automatic Object Detection & Counting)
 * ------------------------------------------------------------
 *  适用场景：
 *    · 浅色背景 + 深色物体（如白底上的黑色回形针）
 *    · 深色背景 + 浅色物体（如暗底上的亮硬币）
 *    程序自动判断目标比背景亮还是暗，无需人工指定极性。
 *
 *  算法流程：
 *    1. 读入图像并转灰度
 *    2. CLAHE 局部对比度增强，让弱对比目标（如发暗的旧硬币）
 *       也能从背景中分离出来
 *    3. 双极性显著性：分别计算
 *          tophat   = gray - open(gray)     突出"比背景亮"的物体
 *          blackhat = close(gray) - gray    突出"比背景暗"的物体
 *       再逐像素取二者最大值，一次覆盖两种极性
 *    4. OTSU 自动阈值二值化，得到前景掩码
 *    5. 形态学开运算去噪点、闭运算填补细缝，使每个物体成为整体
 *    6. findContours 提取外轮廓
 *    7. 面积 + 最小外接矩形长宽比 + 填充度三重过滤，剔除伪目标
 *    8. 绘制旋转外接框与编号，输出计数结果与标注图像
 *
 *  中文路径处理：
 *    Windows 控制台程序默认使用 ANSI 代码页，中文文件名在 main()
 *    的 char* 参数中会乱码。本程序使用 wmain() 接收宽字符路径，
 *    并直接以宽字符构造文件流，因此任意中文文件名均可正常读写。
 *
 *  编译（MSVC，示例）：
 *    call vcvars64.bat
 *    cl /EHsc /std:c++17 /utf-8 /O2 main.cpp /I"<opencv>\build\include" ^
 *       /link /LIBPATH:"<opencv>\build\x64\vc15\lib" opencv_world460.lib
 *
 *  运行：
 *    obj_count.exe [图片1] [图片2] ... [nogui]
 *    不带图片参数时，自动处理当前目录下所有 png/jpg/bmp/tif 图片
 *    加 nogui 表示不弹窗，只保存结果图
 * ============================================================
 */

#define NOMINMAX
#include <windows.h>

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

/* ------------------------------------------------------------
 *  宽字符 <-> UTF-8 辅助函数（用于控制台输出）
 * ---------------------------------------------------------- */
static string wideToUtf8(const wstring& w)
{
    if (w.empty())
        return string();

    int len = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()),
                                  nullptr, 0, nullptr, nullptr);
    string s(static_cast<size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()),
                        s.data(), len, nullptr, nullptr);
    return s;
}

/* ------------------------------------------------------------
 *  宽字符路径读写图像
 *  OpenCV 的 imread/imwrite 只接受窄字符路径，中文路径会失败。
 *  这里"读字节流 -> imdecode / imencode -> 写字节流"，
 *  绕开文件名字符集问题，同时保持 OpenCV 的解码能力。
 * ---------------------------------------------------------- */
static Mat imreadWide(const wstring& path, int flags = IMREAD_COLOR)
{
    try
    {
        ifstream fin(path.c_str(), ios::binary);
        if (!fin)
            return Mat();

        vector<uchar> buf((istreambuf_iterator<char>(fin)), istreambuf_iterator<char>());
        if (buf.empty())
            return Mat();

        return imdecode(buf, flags);
    }
    catch (const exception& e)
    {
        cerr << "读取图像字节流失败：" << e.what() << endl;
        return Mat();
    }
}

static bool imwriteWide(const wstring& path, const Mat& img)
{
    // 由扩展名决定编码格式（只处理常见图片格式）
    string extA = ".png";
    wstring ext = fs::path(path).extension().wstring();
    for (auto& ch : ext)
        ch = static_cast<wchar_t>(towlower(ch));
    if (ext == L".jpg" || ext == L".jpeg")
        extA = ".jpg";
    else if (ext == L".bmp")
        extA = ".bmp";

    vector<uchar> buf;
    if (!imencode(extA, img, buf))
        return false;

    ofstream fout(path.c_str(), ios::binary);
    if (!fout)
        return false;

    fout.write(reinterpret_cast<const char*>(buf.data()), static_cast<streamsize>(buf.size()));
    return fout.good();
}

/* ------------------------------------------------------------
 *  检测参数：集中管理，便于针对不同图片调整
 * ---------------------------------------------------------- */
struct DetectParams
{
    int    openKernel     = 3;      // 开运算核尺寸，去掉零散噪声
    int    closeKernel    = 9;      // 闭运算核尺寸，填补细缝与断点
    double minAreaRatio   = 0.003;  // 最小面积占比（相对整幅图）
    double maxAreaRatio   = 0.150;  // 最大面积占比
    float  minAspect      = 1.0f;   // 最小外接矩形长宽比下限（圆形约等于 1.0）
    float  maxAspect      = 6.0f;   // 长宽比上限，排除贴近边界的细长暗角条
    float  minFill        = 0.25f;  // 轮廓面积 / 外接矩形面积 下限
    float  maxFill        = 1.00f;  // 填充度上限
    float  minCircularity = 0.0f;   // 圆度下限（4*pi*A/P^2），0 表示不检查
    bool   rejectBorder   = false;  // 是否排除接触图像边界（被截断）的目标
};

/* 默认模式：散落物体计数（回形针、硬币等） */
static DetectParams paramsForObjects()
{
    return DetectParams();
}

/* 孔检测模式：只保留"完整的圆孔"
 * 判据比默认模式严格得多：
 *   · 面积上限压到 3%，排除大块暗区（阴影、中心孔等大面积区域）
 *   · 长宽比 1.0~1.6，圆度 >= 0.7，排除刻痕、刻字、条纹等非圆目标
 *   · 填充度 >= 0.55，孔洞是近圆盘形，填充度接近 pi/4 ≈ 0.785
 *   · rejectBorder = true，被画面边缘截断的孔不算"完整"，直接剔除 */
static DetectParams paramsForHoles()
{
    DetectParams p;
    p.minAreaRatio   = 0.002;
    p.maxAreaRatio   = 0.030;
    p.minAspect      = 1.0f;
    p.maxAspect      = 1.6f;
    p.minFill        = 0.55f;
    p.maxFill        = 1.00f;
    p.minCircularity = 0.70f;
    p.rejectBorder   = true;
    p.closeKernel    = 7;
    return p;
}

/* ------------------------------------------------------------
 *  前景掩码：OTSU 自动阈值 + 自动极性判断
 *  OTSU 在"背景 + 物体"双峰直方图上能自动找到最佳分割阈值，
 *  但分不出哪一类是物体。散落物体必然只占少数面积，
 *  因此当白色区域超过一半时，说明被分割出来的是背景，取反即可。
 *  这样同一个程序同时支持"浅底深物"与"深底亮物"。
 * ---------------------------------------------------------- */
static Mat makeForegroundMask(const Mat& gray, const DetectParams& p,
                              double* outThresh = nullptr, double* outRatio = nullptr,
                              bool* outInverted = nullptr)
{
    // 1) OTSU 自动阈值二值化
    Mat mask;
    double t = threshold(gray, mask, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // 2) 自动极性：白色占比超过一半 -> 分割出的是背景 -> 取反
    double fgRatio = static_cast<double>(countNonZero(mask)) / static_cast<double>(mask.total());
    bool inverted = false;
    if (fgRatio > 0.5)
    {
        bitwise_not(mask, mask);
        fgRatio = 1.0 - fgRatio;
        inverted = true;
    }

    if (outThresh)
        *outThresh = t;
    if (outRatio)
        *outRatio = fgRatio;
    if (outInverted)
        *outInverted = inverted;

    // 3) 开运算去掉零散噪点，闭运算把细缝、断点连成整体
    Mat kOpen = getStructuringElement(MORPH_ELLIPSE, Size(p.openKernel, p.openKernel));
    morphologyEx(mask, mask, MORPH_OPEN, kOpen);

    Mat kClose = getStructuringElement(MORPH_ELLIPSE, Size(p.closeKernel, p.closeKernel));
    morphologyEx(mask, mask, MORPH_CLOSE, kClose);

    (void)inverted;
    return mask;
}

/* ------------------------------------------------------------
 *  单个轮廓是否像一个目标物体：面积、长宽比、填充度、圆度、
 *  以及"是否被画面边缘截断"五项校验
 * ---------------------------------------------------------- */
static bool isObjectLike(const vector<Point>& contour, const Size& imgSize, const DetectParams& p)
{
    double area = contourArea(contour);
    double totalArea = static_cast<double>(imgSize.width) * imgSize.height;

    // 面积过滤：太小是噪声，太大是粘连、多物体或大面积暗区
    if (area < totalArea * p.minAreaRatio || area > totalArea * p.maxAreaRatio)
        return false;

    RotatedRect rr = minAreaRect(contour);
    float w = rr.size.width;
    float h = rr.size.height;
    if (w <= 1.0f || h <= 1.0f)
        return false;

    // 长宽比过滤：圆形目标接近 1，细长目标可到 6
    float longSide = max(w, h);
    float shortSide = min(w, h);
    float aspect = longSide / shortSide;
    if (aspect < p.minAspect || aspect > p.maxAspect)
        return false;

    // 填充度过滤：轮廓面积占外接矩形的比例
    float fill = static_cast<float>(area) / (w * h);
    if (fill < p.minFill || fill > p.maxFill)
        return false;

    // 圆度过滤：4*pi*A/P^2，正圆 = 1.0，越不规则越小
    if (p.minCircularity > 0.0f)
    {
        double perimeter = arcLength(contour, true);
        if (perimeter <= 1e-6)
            return false;
        float circularity = static_cast<float>(4.0 * CV_PI * area / (perimeter * perimeter));
        if (circularity < p.minCircularity)
            return false;
    }

    // 完整性过滤：外接矩形接触画面边缘说明目标被截断，不算完整目标
    if (p.rejectBorder)
    {
        Rect br = boundingRect(contour);
        const int margin = 2;
        if (br.x <= margin || br.y <= margin ||
            br.x + br.width >= imgSize.width - margin ||
            br.y + br.height >= imgSize.height - margin)
            return false;
    }

    return true;
}

/* ------------------------------------------------------------
 *  判断文件名是否为本程序的输出结果（避免重复处理）
 * ---------------------------------------------------------- */
static bool isResultStem(const wstring& stem)
{
    auto endsWith = [](const wstring& s, const wstring& suffix) -> bool
    {
        if (s.size() < suffix.size())
            return false;
        return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    return endsWith(stem, L"_counted") || endsWith(stem, L"_mask");
}

/* ------------------------------------------------------------
 *  处理单张图片：返回检测数量，失败返回 -1
 * ---------------------------------------------------------- */
static int processImage(const wstring& imgPath, bool showGui, const DetectParams& params)
{
    Mat src = imreadWide(imgPath, IMREAD_COLOR);
    if (src.empty())
    {
        cerr << "错误：无法读取图像，请检查路径：" << wideToUtf8(imgPath) << endl;
        return -1;
    }
    cout << "\n已读入图像：" << wideToUtf8(imgPath)
         << "  (" << src.cols << " x " << src.rows << ")" << endl;

    /* ---------- 灰度化与降噪 ---------- */
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(3, 3), 0);

    /* ---------- 前景掩码 ---------- */
    double thresh = 0, fgRatio = 0;
    bool inverted = false;
    Mat mask = makeForegroundMask(gray, params, &thresh, &fgRatio, &inverted);
    cout << "OTSU 阈值：" << thresh << "，目标占比：" << fgRatio * 100.0 << "%"
         << (inverted ? "（目标比背景暗，已取反）" : "（目标比背景亮）") << endl;

    /* ---------- 轮廓提取与过滤 ---------- */
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<RotatedRect> targets;
    for (size_t i = 0; i < contours.size(); ++i)
    {
        if (!isObjectLike(contours[i], src.size(), params))
            continue;
        targets.push_back(minAreaRect(contours[i]));
    }

    int count = static_cast<int>(targets.size());
    cout << "检测到目标数量：" << count << endl;

    /* ---------- 结果可视化 ---------- */
    Mat vis = src.clone();
    const Scalar boxColor(0, 255, 0);    // 绿色旋转框
    const Scalar textColor(0, 0, 255);   // 红色编号
    const Scalar tagBg(0, 0, 0);         // 左上角标签底色
    const Scalar tagFg(255, 255, 255);   // 左上角标签文字

    double fontScale = max(0.5, src.cols / 800.0);

    for (int i = 0; i < count; ++i)
    {
        RotatedRect rr = targets[i];

        // 旋转外接矩形：比正矩形更贴合倾斜摆放的目标
        Point2f pts[4];
        rr.points(pts);
        for (int k = 0; k < 4; ++k)
            line(vis, pts[k], pts[(k + 1) % 4], boxColor, 2, LINE_AA);

        // 编号写在矩形中心
        char label[8];
        snprintf(label, sizeof(label), "%d", i + 1);
        int baseLine = 0;
        double fs2 = 0.6 * fontScale;
        Size ts = getTextSize(label, FONT_HERSHEY_SIMPLEX, fs2, 2, &baseLine);
        Point org(static_cast<int>(rr.center.x - ts.width / 2.0),
                  static_cast<int>(rr.center.y + ts.height / 2.0));
        putText(vis, label, org, FONT_HERSHEY_SIMPLEX, fs2, textColor, 2, LINE_AA);
    }

    // 左上角显示总数
    char summary[64];
    snprintf(summary, sizeof(summary), "Count = %d", count);
    int baseLine2 = 0;
    double fs3 = 1.0 * fontScale;
    Size ts2 = getTextSize(summary, FONT_HERSHEY_SIMPLEX, fs3, 2, &baseLine2);
    rectangle(vis, Point(6, 6), Point(14 + ts2.width, 14 + ts2.height + baseLine2),
              tagBg, FILLED);
    putText(vis, summary, Point(10, 10 + ts2.height),
            FONT_HERSHEY_SIMPLEX, fs3, tagFg, 2, LINE_AA);

    /* ---------- 保存结果 ---------- */
    fs::path imgFile(imgPath);
    fs::path outDir = imgFile.parent_path();
    if (outDir.empty())
        outDir = fs::current_path();
    wstring stem = imgFile.stem().wstring();

    wstring visPath = (outDir / (stem + L"_counted.png")).wstring();
    wstring maskPath = (outDir / (stem + L"_mask.png")).wstring();

    bool okVis = imwriteWide(visPath, vis);
    bool okMask = imwriteWide(maskPath, mask);

    cout << "标注结果图：" << wideToUtf8(visPath)
         << (okVis ? "  [已保存]" : "  [保存失败]") << endl;
    cout << "二值掩码图：" << wideToUtf8(maskPath)
         << (okMask ? "  [已保存]" : "  [保存失败]") << endl;

    /* ---------- 交互显示（可选） ---------- */
    if (showGui)
    {
        imshow("Counting Result", vis);
        imshow("Foreground Mask", mask);
        cout << "按任意键继续..." << endl;
        waitKey(0);
        destroyAllWindows();
    }

    return count;
}

/* ------------------------------------------------------------
 *  主流程
 * ---------------------------------------------------------- */
int wmain(int argc, wchar_t** argv)
{
    // 让控制台按 UTF-8 显示，中文路径与提示不会乱码
    SetConsoleOutputCP(CP_UTF8);

    try
    {
        /* ---------- 命令行参数 ---------- */
        bool showGui = true;
        bool holesMode = false;
        vector<wstring> inputs;
        for (int i = 1; i < argc; ++i)
        {
            wstring a = argv[i];
            if (a == L"nogui")
            {
                showGui = false;
                continue;
            }
            if (a == L"holes")
            {
                holesMode = true;
                continue;
            }
            inputs.push_back(a);
        }

        const DetectParams params = holesMode ? paramsForHoles() : paramsForObjects();
        cout << "检测模式：" << (holesMode ? "孔检测（只统计完整的圆孔）" : "通用物体计数") << endl;
        fs::path curDir = fs::current_path();

        // 未指定图片时，自动收集当前目录下的图片（排除本程序输出的结果图）
        if (inputs.empty())
        {
            set<wstring> exts = { L".png", L".jpg", L".jpeg", L".bmp", L".tif", L".tiff" };
            for (const auto& entry : fs::directory_iterator(curDir))
            {
                if (!entry.is_regular_file())
                    continue;

                wstring ext = entry.path().extension().wstring();
                for (auto& ch : ext)
                    ch = static_cast<wchar_t>(towlower(ch));
                if (exts.find(ext) == exts.end())
                    continue;

                if (isResultStem(entry.path().stem().wstring()))
                    continue;

                inputs.push_back(entry.path().wstring());
            }

            if (inputs.empty())
            {
                cerr << "错误：当前目录下没有找到待检测的图片。" << endl;
                return -1;
            }
            sort(inputs.begin(), inputs.end());
            cout << "未指定图片，自动处理当前目录下 " << inputs.size() << " 张图片。" << endl;
        }

        /* ---------- 逐张处理 ---------- */
        int okCount = 0;
        int failCount = 0;
        vector<pair<wstring, int>> results;

        for (const auto& imgPath : inputs)
        {
            int n = processImage(imgPath, showGui, params);
            if (n < 0)
            {
                ++failCount;
                continue;
            }
            ++okCount;
            results.push_back(make_pair(imgPath, n));
        }

        /* ---------- 汇总 ---------- */
        cout << "\n================ 检测汇总 ================" << endl;
        for (size_t i = 0; i < results.size(); ++i)
        {
            cout << "  " << wideToUtf8(fs::path(results[i].first).filename().wstring())
                 << "  ->  " << results[i].second << " 个" << endl;
        }
        cout << "成功 " << okCount << " 张，失败 " << failCount << " 张" << endl;

        return failCount > 0 ? -1 : 0;
    }
    catch (const cv::Exception& e)
    {
        cerr << "OpenCV 异常：" << e.what() << endl;
        return -2;
    }
    catch (const std::exception& e)
    {
        cerr << "运行异常：" << e.what() << endl;
        return -3;
    }
}

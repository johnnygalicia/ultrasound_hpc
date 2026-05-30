#include <opencv2/opencv.hpp>

#include <filesystem>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>

#include <omp.h>

namespace fs = std::filesystem;

int main()
{
    // =====================================================
    // PATHS
    // =====================================================

    std::string input_dir =
        "/mnt/c/Users/Johnny/Documents/"
        "escuela_semestre2026-2/TSCAD/"
        "proyecto_final/ultrasound_hpc/"
        "data/processed/video_2_frames";

    std::string output_dir =
        "/mnt/c/Users/Johnny/Documents/"
        "escuela_semestre2026-2/TSCAD/"
        "proyecto_final/ultrasound_hpc/"
        "data/processed/video2_bilateral_sharpen_multi_thread";

    // =====================================================
    // CREATE OUTPUT DIRECTORY
    // =====================================================

    fs::create_directories(output_dir);

    // =====================================================
    // GET FRAME LIST
    // =====================================================

    std::vector<std::string> frame_paths;

    for (const auto& entry :
         fs::directory_iterator(input_dir))
    {
        if (entry.path().extension() == ".png")
        {
            frame_paths.push_back(
                entry.path().string()
            );
        }
    }

    std::sort(
        frame_paths.begin(),
        frame_paths.end()
    );

    std::cout << "\n=====================================\n";

    std::cout << "OPENMP BILATERAL + SHARPEN PIPELINE\n";

    std::cout << "=====================================\n\n";

    std::cout << "Frames encontrados: "
              << frame_paths.size()
              << "\n\n";

    // =====================================================
    // SHARPEN KERNEL
    // =====================================================

    cv::Mat sharpen_kernel =
        (cv::Mat_<float>(3,3)
        <<
            0, -1,  0,
           -1,  5, -1,
            0, -1,  0);

    // =====================================================
    // TIMER START
    // =====================================================

    auto start =
        std::chrono::high_resolution_clock::now();

    // =====================================================
    // OPENMP PARALLEL LOOP
    // =====================================================

    #pragma omp parallel for

    for (int i = 0;
         i < frame_paths.size();
         i++)
    {
        // =============================================
        // LOAD IMAGE
        // =============================================

        cv::Mat image =
            cv::imread(
                frame_paths[i],
                cv::IMREAD_GRAYSCALE
            );

        // =============================================
        // STEP 1:
        // BILATERAL FILTER
        // =============================================

        cv::Mat bilateral_result;

        cv::bilateralFilter(
            image,
            bilateral_result,
            5,
            25,
            10
        );

        // =============================================
        // STEP 2:
        // SHARPEN FILTER
        // =============================================

        cv::Mat sharpen_result;

        cv::filter2D(
            bilateral_result,
            sharpen_result,
            -1,
            sharpen_kernel
        );

        // =============================================
        // OUTPUT PATH
        // =============================================

        std::string filename =
            fs::path(frame_paths[i])
            .filename()
            .string();

        std::string output_path =
            output_dir + "/" + filename;

        // =============================================
        // SAVE IMAGE
        // =============================================

        cv::imwrite(
            output_path,
            sharpen_result
        );
    }

    // =====================================================
    // TIMER END
    // =====================================================

    auto end =
        std::chrono::high_resolution_clock::now();

    double elapsed =
        std::chrono::duration<double>(
            end - start
        ).count();

    // =====================================================
    // METRICS
    // =====================================================

    double fps =
        frame_paths.size() / elapsed;

    double avg_time =
        elapsed / frame_paths.size();

    // =====================================================
    // RESULTS
    // =====================================================

    std::cout << "\n=====================================\n";

    std::cout << "RESULTS\n";

    std::cout << "=====================================\n\n";

    std::cout << "Tiempo total: "
              << elapsed
              << " s\n";

    std::cout << "Tiempo/frame: "
              << avg_time
              << " s\n";

    std::cout << "FPS: "
              << fps
              << "\n";

    std::cout << "\nFrames guardados en:\n";

    std::cout << output_dir << "\n";

    return 0;
}
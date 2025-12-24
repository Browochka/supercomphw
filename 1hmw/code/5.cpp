#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <random>
#include <limits>
#include <sstream>
#include <algorithm>
#include <sys/stat.h>
#include <unistd.h>

bool directory_exists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

bool create_directory(const std::string& path) {
    return mkdir(path.c_str(), 0755) == 0;
}

void compute_max_of_mins(const std::vector<std::vector<int>>& matrix, int num_threads, const std::string& schedule_str)
{
    omp_set_num_threads(num_threads);

    omp_sched_t sched;
    int chunk_size = 0;
    if (schedule_str == "dynamic") {
        sched = omp_sched_dynamic;
        chunk_size = 10;
    }
    else if (schedule_str == "guided") {
        sched = omp_sched_guided;
    }
    else {
        sched = omp_sched_static;
    }
    omp_set_schedule(sched, chunk_size);

    int max_of_mins = std::numeric_limits<int>::min();

    #pragma omp parallel for schedule(runtime) reduction(max:max_of_mins)
    for (int i = 0; i < matrix.size(); ++i) {
        int min_in_row = std::numeric_limits<int>::max();
        for (int val : matrix[i]) {
            if (val < min_in_row) min_in_row = val;
        }
        if (min_in_row > max_of_mins) max_of_mins = min_in_row;
    }
}

std::vector<std::vector<int>> generate_banded(size_t n, int k, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-10000, 10000);

    std::vector<std::vector<int>> mat(n, std::vector<int>(n, std::numeric_limits<int>::max()));
    for (size_t i = 0; i < n; ++i) {
        int start = std::max(0, static_cast<int>(i) - k);
        int end = std::min(static_cast<int>(n) - 1, static_cast<int>(i) + k);
        for (int j = start; j <= end; ++j) {
            mat[i][j] = dist(rng);
        }
    }
    return mat;
}

std::vector<std::vector<int>> generate_lower_triangular(size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-10000, 10000);

    std::vector<std::vector<int>> mat(n, std::vector<int>(n, std::numeric_limits<int>::max()));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            mat[i][j] = dist(rng);
        }
    }
    return mat;
}

int main()
{
    std::cout << " Запуск программы для тестирования разных типов матриц и стратегий планирования...\n";

    const std::vector<int> sizes = { 1000, 3000, 5000 };
    const std::vector<std::string> matrix_types = { "banded", "lower" };
    // Базовый список, который будет отфильтрован
    const std::vector<int> thread_counts_all = { 1, 2, 4, 6, 8, 12, 16, 32 };
    const std::vector<std::string> schedules = { "static", "dynamic", "guided" };

    // Фильтруем потокы: оставляем только те, что <= 12
    std::vector<int> thread_counts;
    const int MAX_THREADS = 12;
    for (int t : thread_counts_all) {
        if (t <= MAX_THREADS) {
            thread_counts.push_back(t);
        }
    }

    std::cout << " Будут протестированы следующие количества потоков: ";
    for (int t : thread_counts) std::cout << t << " ";
    std::cout << "\n";

    std::string results_dir = "./Results";

    std::cout << " Проверяем наличие директории '" << results_dir << "'...\n";
    if (!directory_exists(results_dir)) {
        std::cout << " Директория не найдена. Создаем...\n";
        if (!create_directory(results_dir)) {
            std::cerr << " Ошибка: Не удалось создать директорию Results!\n";
            return 1;
        }
        std::cout << " Директория успешно создана.\n";
    } else {
        std::cout << " Директория уже существует.\n";
    }

    std::string log_path = results_dir + "/5_log.txt";
    std::ofstream log_file(log_path);

    if (!log_file.is_open()) {
        std::cerr << " Ошибка: Не удалось открыть файл для записи!\n";
        return 1;
    }

    std::cout << " Файл для записи результатов открыт: " << log_path << "\n\n";

    const int num_tests = 3;
    const unsigned seed = 42;

    for (const auto& type : matrix_types) {
        std::cout << "==============================\n";
        std::cout << " Начинаем тестирование для типа матрицы: " << type << "\n";

        for (const int n : sizes) {
            std::cout << "--------------------------------------\n";
            std::cout << " Размер матрицы: " << n << "x" << n << " ("
                      << (static_cast<long long>(n) * n) << " элементов)\n";

            std::vector<std::vector<int>> matrix;
            int k = 0;

            std::cout << "    Генерация матрицы... ";
            if (type == "banded") {
                k = n / 10;
                matrix = generate_banded(n, k, seed);
                std::cout << "ленточная (k=" << k << ")\n";
            }
            else if (type == "lower") {
                matrix = generate_lower_triangular(n, seed);
                std::cout << "нижняя треугольная\n";
            }

            double base_time = 0.0;
            {
                std::cout << "    Базовый замер (1 поток, static schedule)... ";
                double total = 0.0;
                for (int t = 0; t < num_tests; ++t) {
                    const auto start = std::chrono::high_resolution_clock::now();
                    compute_max_of_mins(matrix, 1, "static");
                    const auto end = std::chrono::high_resolution_clock::now();
                    total += std::chrono::duration<double, std::milli>(end - start).count();
                }
                base_time = total / num_tests;
                std::cout << base_time << " мс\n";
            }

            for (const auto& schedule : schedules) {
                std::cout << "   Стратегия планирования: " << schedule << "\n";

                log_file << "Size: " << n << "x" << n << ", elements = "
                         << (static_cast<long long>(n) * n)
                         << ", Matrix type: " << type
                         << ", Schedule: " << schedule << "\n";

                log_file << "Threads: 1\n";
                log_file << "  Time: " << base_time << " ms (speedup: 1x, efficiency: 1)\n";

                for (int threads : thread_counts) {
                    if (threads == 1) continue;

                    std::cout << "       Потоков: " << threads << "... ";
                    double total = 0.0;
                    for (int t = 0; t < num_tests; ++t) {
                        const auto start = std::chrono::high_resolution_clock::now();
                        compute_max_of_mins(matrix, threads, schedule);
                        const auto end = std::chrono::high_resolution_clock::now();
                        total += std::chrono::duration<double, std::milli>(end - start).count();
                    }
                    const double avg_time = total / num_tests;
                    const double speedup = base_time / avg_time;
                    const double efficiency = speedup / threads;

                    log_file << "Threads: " << threads << "\n";
                    log_file << "  Time: " << avg_time << " ms (speedup: "
                             << speedup << "x, efficiency: " << efficiency << ")\n";

                    std::cout << avg_time << " мс (ускорение: " << speedup << "x)\n";
                }
                log_file << "--------------------------------------\n";
                std::cout << "\n";
            }
            std::cout << " Тестирование для размера " << n << "x" << n << " завершено.\n";
        }
        std::cout << "Тестирование для типа '" << type << "' завершено.\n";
    }

    log_file.close();

    std::cout << "\n======================================\n";
    std::cout << " Все тесты завершены. Результаты сохранены в файл: " << log_path << "\n";
    std::cout << " Программа успешно завершена!\n";

    return 0;
}

#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <random>
#include <limits>
#include <sys/stat.h>

void compute_max_of_mins(const std::vector<std::vector<int>>& matrix, int num_threads) {
    omp_set_num_threads(num_threads);

    int max_of_mins = std::numeric_limits<int>::min();

    #pragma omp parallel for reduction(max:max_of_mins)
    for (int i = 0; i < matrix.size(); ++i) {
        int min_in_row = std::numeric_limits<int>::max();
        for (int val : matrix[i]) {
            if (val < min_in_row) min_in_row = val;
        }
        if (min_in_row > max_of_mins) max_of_mins = min_in_row;
    }
}

std::vector<std::vector<int>> generate_matrix(size_t rows, size_t cols, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-10000, 10000);

    std::vector<std::vector<int>> mat(rows, std::vector<int>(cols));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            mat[i][j] = dist(rng);

    return mat;
}

bool directory_exists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

bool create_directory(const std::string& path) {
    return mkdir(path.c_str(), 0755) == 0;
}

int main() {
    std::cout << "🔄 Начинаем вычисление максимума из минимумов строк матрицы..." << std::endl;
    
    const std::vector<std::pair<int, int>> sizes = {
        {1000, 1000},
        {5000, 5000},
        {10000, 10000},
        {100000, 10000}
    };

    const int MAX_THREADS = 12;
    std::vector<int> thread_counts;
    
    for (int t : {1, 2, 4, 6, 8, 12}) {
        if (t <= MAX_THREADS) {
            thread_counts.push_back(t);
        }
    }
    
    std::cout << " Тестируемые количества потоков: ";
    for (int t : thread_counts) std::cout << t << " ";
    std::cout << "(максимум " << MAX_THREADS << ")" << std::endl;

    std::string results_dir = "./Results";
    
    std::cout << " Проверяем наличие директории Results..." << std::endl;
    if (!directory_exists(results_dir)) {
        std::cout << " Создаем директорию Results..." << std::endl;
        if (!create_directory(results_dir)) {
            std::cerr << " Ошибка: Не удалось создать директорию Results!\n";
            return 1;
        }
        std::cout << " Директория Results создана успешно" << std::endl;
    } else {
        std::cout << "Директория Results уже существует" << std::endl;
    }

    std::string log_path = results_dir + "/4_log.txt";
    std::ofstream log_file(log_path);
    
    if (!log_file.is_open()) {
        std::cerr << " Ошибка: Не удалось открыть файл для записи!\n";
        return 1;
    }
    
    std::cout << " Файл для записи результатов открыт: " << log_path << std::endl;

    const int num_tests = 3;
    double base_time = 0.0;
    const unsigned seed = 42;
    
    log_file << "Max threads limited to: " << MAX_THREADS << "\n";
    log_file << "Threads tested: ";
    for (int t : thread_counts) log_file << t << " ";
    log_file << "\n";
    log_file << "Matrix sizes tested:\n";
    for (const auto& p : sizes) {
        log_file << "  " << p.first << "x" << p.second
                 << " (" << (static_cast<long long>(p.first) * p.second) << " elements)\n";
    }
    log_file << "--------------------------------------\n";

    for (const auto& p : sizes) {
        const int rows = p.first;
        const int cols = p.second;
        long long total_elements = static_cast<long long>(rows) * cols;
        
        std::cout << "\n🔧 Обрабатываем матрицу " << rows << "x" << cols
                  << " (" << total_elements << " элементов)..." << std::endl;
        
        log_file << "Matrix: rows = " << rows << ", cols = " << cols
                 << ", elements = " << total_elements << "\n";

        std::cout << "    Генерируем матрицу..." << std::endl;
        auto matrix = generate_matrix(rows, cols, seed);
        std::cout << "    Матрица сгенерирована" << std::endl;

        {
            std::cout << "    Выполняем базовый замер (1 поток)..." << std::endl;
            double total = 0.0;
            for (int t = 0; t < num_tests; ++t) {
                const auto start = std::chrono::high_resolution_clock::now();
                compute_max_of_mins(matrix, 1);
                const auto end = std::chrono::high_resolution_clock::now();
                total += std::chrono::duration<double, std::milli>(end - start).count();
            }
            base_time = total / num_tests;
            log_file << "Threads: 1\n";
            log_file << "  Time: " << base_time << " ms (speedup: 1x, efficiency: 1)\n";
            std::cout << "   Базовый замер завершен: " << base_time << " мс" << std::endl;
        }

        std::cout << "   Начинаем тестирование с разным количеством потоков..." << std::endl;
        for (int threads : thread_counts) {
            if (threads == 1) continue;
            
            if (threads > MAX_THREADS) {
                std::cout << "     Пропускаем " << threads << " потоков (превышает лимит "
                          << MAX_THREADS << ")" << std::endl;
                continue;
            }

            std::cout << "  Тестируем " << threads << " потоков..." << std::endl;

            double total = 0.0;
            for (int t = 0; t < num_tests; ++t) {
                const auto start = std::chrono::high_resolution_clock::now();
                compute_max_of_mins(matrix, threads);
                const auto end = std::chrono::high_resolution_clock::now();
                total += std::chrono::duration<double, std::milli>(end - start).count();
            }
            const double avg_time = total / num_tests;
            const double speedup = base_time / avg_time;
            const double efficiency = speedup / threads;

            log_file << "Threads: " << threads << "\n";
            log_file << "  Time: " << avg_time << " ms (speedup: " << speedup
                     << "x, efficiency: " << efficiency << ")" << "\n";
            
            std::cout << "  " << threads << " потоков: "
                      << avg_time << " мс (ускорение: " << speedup << "x)" << std::endl;
        }
        log_file << "--------------------------------------\n";
        std::cout << " Матрица " << rows << "x" << cols << " полностью обработана" << std::endl;
    }

    log_file.close();
    std::cout << "\n======================================" << std::endl;
    std::cout << " Результаты сохранены в файл: " << log_path << std::endl;
    std::cout << " Программа завершена успешно!" << std::endl;
    
    return 0;
}

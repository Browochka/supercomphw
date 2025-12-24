#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <random>
#include <sys/stat.h>

void test_reduction_method(const std::vector<double>& a, int num_threads, const std::string& method) {
    omp_set_num_threads(num_threads);

    double sum = 0.0;

    if (method == "reduction") {
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < a.size(); ++i) {
            sum += a[i];
        }
    }
    else if (method == "atomic") {
        #pragma omp parallel for
        for (size_t i = 0; i < a.size(); ++i) {
            #pragma omp atomic
            sum += a[i];
        }
    }
    else if (method == "critical") {
        #pragma omp parallel for
        for (size_t i = 0; i < a.size(); ++i) {
            #pragma omp critical
            sum += a[i];
        }
    }
    else if (method == "lock") {
        omp_lock_t lock;
        omp_init_lock(&lock);
        #pragma omp parallel for
        for (size_t i = 0; i < a.size(); ++i) {
            omp_set_lock(&lock);
            sum += a[i];
            omp_unset_lock(&lock);
        }
        omp_destroy_lock(&lock);
    }
}

bool directory_exists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

bool create_directory(const std::string& path) {
    return mkdir(path.c_str(), 0755) == 0;
}

int main() {
    std::cout << "Начинаем тестирование методов редукции в OpenMP" << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1000.0);

    std::vector<int> thread_counts = { 1, 2, 4, 6, 8, 12 };
    std::vector<size_t> sizes = { 500000, 1000000, 5000000, 10000000 };
    std::vector<std::string> methods = { "reduction", "atomic", "critical", "lock" };

    std::cout << "Тестируемые количества потоков: ";
    for (int t : thread_counts) std::cout << t << " ";
    std::cout << std::endl;
    
    std::cout << "Тестируемые размеры векторов: ";
    for (size_t s : sizes) std::cout << s << " ";
    std::cout << std::endl;
    
    std::cout << "Тестируемые методы редукции: ";
    for (const auto& m : methods) std::cout << m << " ";
    std::cout << std::endl;

    std::string results_dir = "./Results";
    
    std::cout << "Проверяем наличие директории Results" << std::endl;
    if (!directory_exists(results_dir)) {
        std::cout << "Создаем директорию Results" << std::endl;
        if (!create_directory(results_dir)) {
            std::cerr << "Ошибка: Не удалось создать директорию Results" << std::endl;
            return 1;
        }
        std::cout << "Директория Results создана успешно" << std::endl;
    } else {
        std::cout << "Директория Results уже существует" << std::endl;
    }

    std::string log_path = results_dir + "/7_log.txt";
    std::ofstream log_file(log_path);
    
    if (!log_file.is_open()) {
        std::cerr << "Ошибка: Не удалось открыть файл для записи" << std::endl;
        return 1;
    }
    
    std::cout << "Файл для записи результатов открыт: " << log_path << std::endl;

    const int num_tests = 3;
    double base_time = 0.0;

    log_file << "OpenMP Reduction Methods Testing\n";
    log_file << "Threads tested: ";
    for (int t : thread_counts) log_file << t << " ";
    log_file << "\nVector sizes: ";
    for (size_t s : sizes) log_file << s << " ";
    log_file << "\nMethods: ";
    for (const auto& m : methods) log_file << m << " ";
    log_file << "\n";
    log_file << "--------------------------------------\n";

    for (size_t size : sizes) {
        std::cout << "Работа с вектором размером: " << size << std::endl;
        
        std::cout << "Генерируем случайные данные" << std::endl;
        std::vector<double> a(size);
        for (size_t i = 0; i < size; ++i)
            a[i] = dist(gen);
        std::cout << "Данные сгенерированы" << std::endl;

        for (const auto& method : methods) {
            std::cout << "Тестируем метод: " << method << std::endl;
            
            log_file << "Vector size: " << size << "\n";
            log_file << "Method: " << method << "\n";

            std::cout << "Базовый замер (1 поток)" << std::endl;
            double total_time = 0.0;
            for (int t = 0; t < num_tests; ++t) {
                auto start = std::chrono::high_resolution_clock::now();
                test_reduction_method(a, 1, method);
                auto end = std::chrono::high_resolution_clock::now();
                total_time += std::chrono::duration<double, std::milli>(end - start).count();
            }
            base_time = total_time / num_tests;
            log_file << "Threads: 1\n";
            log_file << " Time: " << base_time << " ms (speedup: 1.0x, efficiency: 1.0)\n";
            
            std::cout << "Базовый замер: " << base_time << " мс" << std::endl;

            std::cout << "Начинаем тестирование с разным количеством потоков" << std::endl;
            for (int threads : thread_counts) {
                if (threads == 1) continue;
                
                std::cout << "Тестируем " << threads << " потоков" << std::endl;
                
                total_time = 0.0;
                for (int t = 0; t < num_tests; ++t) {
                    auto start = std::chrono::high_resolution_clock::now();
                    test_reduction_method(a, threads, method);
                    auto end = std::chrono::high_resolution_clock::now();
                    total_time += std::chrono::duration<double, std::milli>(end - start).count();
                }
                double avg_time = total_time / num_tests;
                double speedup = base_time / avg_time;
                double efficiency = speedup / threads;
                log_file << "Threads: " << threads << "\n";
                log_file << " Time: " << avg_time << " ms (speedup: " << speedup << "x, efficiency: " << efficiency << ")\n";
                
                std::cout << threads << " потоков: " << avg_time << " мс (ускорение: " << speedup << "x)" << std::endl;
            }
            log_file << "--------------------------------------\n";
            std::cout << "Метод '" << method << "' протестирован" << std::endl;
        }
        std::cout << "Вектор размером " << size << " полностью обработан" << std::endl;
    }

    log_file.close();
    
    std::cout << "Результаты сохранены в файл: " << log_path << std::endl;
    std::cout << "Программа завершена успешно" << std::endl;
    
    return 0;
}

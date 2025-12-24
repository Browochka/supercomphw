#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <random>
#include <cmath>
#include <sys/stat.h>

void test_schedule(const std::vector<int>& a, int num_threads, const std::string& schedule_type) {
    omp_set_num_threads(num_threads);

    if (schedule_type == "static")
        omp_set_schedule(omp_sched_static, 0);
    else if (schedule_type == "dynamic")
        omp_set_schedule(omp_sched_dynamic, 5);
    else if (schedule_type == "guided")
        omp_set_schedule(omp_sched_guided, 0);

    double sum = 0.0;

    #pragma omp parallel for schedule(runtime) reduction(+:sum)
    for (int i = 0; i < (int)a.size(); ++i) {
        int work = a[i] % 1000;
        double local_sum = 0.0;
        for (int j = 0; j < work; ++j) {
            local_sum += std::sin(j * 0.001);
        }
        sum += local_sum;
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
    std::cout << "Начинаем тестирование стратегий планирования OpenMP" << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 1000);

    std::vector<int> thread_counts = { 1, 2, 4, 6, 8, 12 };
    std::vector<size_t> sizes = { 10000, 100000, 500000 };
    std::vector<std::string> schedules = { "static", "dynamic", "guided" };

    std::cout << "Тестируемые количества потоков: ";
    for (int t : thread_counts) std::cout << t << " ";
    std::cout << std::endl;
    
    std::cout << "Тестируемые размеры векторов: ";
    for (size_t s : sizes) std::cout << s << " ";
    std::cout << std::endl;
    
    std::cout << "Тестируемые стратегии планирования: ";
    for (const auto& s : schedules) std::cout << s << " ";
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

    std::string log_path = results_dir + "/6_log.txt";
    std::ofstream log_file(log_path);
    
    if (!log_file.is_open()) {
        std::cerr << "Ошибка: Не удалось открыть файл для записи" << std::endl;
        return 1;
    }
    
    std::cout << "Файл для записи результатов открыт: " << log_path << std::endl;

    const int num_tests = 3;

    log_file << "OpenMP Schedule Testing\n";
    log_file << "Threads tested: ";
    for (int t : thread_counts) log_file << t << " ";
    log_file << "\nVector sizes: ";
    for (size_t s : sizes) log_file << s << " ";
    log_file << "\nSchedules: ";
    for (const auto& s : schedules) log_file << s << " ";
    log_file << "\n";
    log_file << "--------------------------------------\n";

    for (size_t size : sizes) {
        std::cout << "Работа с вектором размером: " << size << std::endl;
        
        std::cout << "Генерируем случайные данные" << std::endl;
        std::vector<int> a(size);
        for (size_t i = 0; i < size; ++i) {
            a[i] = dist(gen);
        }
        std::cout << "Данные сгенерированы" << std::endl;

        for (const auto& schedule : schedules) {
            std::cout << "Тестируем стратегию: " << schedule << std::endl;
            
            log_file << "Vector size: " << size << "\n";
            log_file << "Schedule: " << schedule << "\n";

            std::cout << "Базовый замер (1 поток)" << std::endl;
            double total_time = 0.0;
            for (int t = 0; t < num_tests; ++t) {
                auto start = std::chrono::high_resolution_clock::now();
                test_schedule(a, 1, schedule);
                auto end = std::chrono::high_resolution_clock::now();
                total_time += std::chrono::duration<double, std::milli>(end - start).count();
            }
            double base_time = total_time / num_tests;
            double speedup = 1.0;
            double efficiency = 1.0;
            
            log_file << "Threads: 1\n";
            log_file << " Time: " << base_time << " ms (speedup: " << speedup << "x, efficiency: " << efficiency << ")\n";
            
            std::cout << "Базовый замер: " << base_time << " мс" << std::endl;

            std::cout << "Начинаем тестирование с разным количеством потоков" << std::endl;
            for (int threads : thread_counts) {
                if (threads == 1) continue;

                std::cout << "Тестируем " << threads << " потоков" << std::endl;
                
                total_time = 0.0;
                for (int t = 0; t < num_tests; ++t) {
                    auto start = std::chrono::high_resolution_clock::now();
                    test_schedule(a, threads, schedule);
                    auto end = std::chrono::high_resolution_clock::now();
                    total_time += std::chrono::duration<double, std::milli>(end - start).count();
                }
                double avg_time = total_time / num_tests;
                speedup = base_time / avg_time;
                efficiency = speedup / threads;
                
                log_file << "Threads: " << threads << "\n";
                log_file << " Time: " << avg_time << " ms (speedup: " << speedup << "x, efficiency: " << efficiency << ")\n";
                
                std::cout << threads << " потоков: " << avg_time << " мс (ускорение: " << speedup << "x)" << std::endl;
            }
            log_file << "--------------------------------------\n";
            std::cout << "Стратегия '" << schedule << "' протестирована" << std::endl;
        }
        std::cout << "Вектор размером " << size << " полностью обработан" << std::endl;
    }

    log_file.close();
    
    std::cout << "Результаты сохранены в файл: " << log_path << std::endl;
    std::cout << "Программа завершена успешно" << std::endl;
    
    return 0;
}

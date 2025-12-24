#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <random>
#include <sys/stat.h>
#include <unistd.h>

void scalar_production(const std::vector<int>& a, const std::vector<int>& b, int num_threads) {
    omp_set_num_threads(num_threads);
    int result = 0;

    #pragma omp parallel for reduction(+:result)
    for (int i = 0; i < (int)a.size(); ++i) {
        result += a[i] * b[i];
    }
}

bool directory_exists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

bool create_directory(const std::string& path) {
    return mkdir(path.c_str(), 0755) == 0;
}

int get_available_processors() {
    return sysconf(_SC_NPROCESSORS_ONLN);
}

int main() {
    std::cout << "Начинаем вычисление скалярного произведения..." << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 1000);

    int max_procs = get_available_processors();
    std::cout << " Доступно процессоров: " << max_procs << std::endl;
    
    std::vector<int> thread_counts;
    for (int t : {1, 2, 4, 6, 8, 12}) {
        if (t <= max_procs * 2) {
            thread_counts.push_back(t);
        }
    }
    
    std::cout << " Тестируемые количества потоков: ";
    for (int t : thread_counts) std::cout << t << " ";
    std::cout << std::endl;
    
    std::vector<size_t> sizes = { 100000, 1000000, 10000000, 50000000 };

    std::string results_dir = "./Results";
    
    std::cout << " Проверяем наличие директории Results..." << std::endl;
    if (!directory_exists(results_dir)) {
        std::cout << " Создаем директорию Results..." << std::endl;
        if (!create_directory(results_dir)) {
            std::cout << "Ошибка: Не удалось создать директорию Results!" << std::endl;
            return 1;
        }
        std::cout << "Директория Results создана успешно" << std::endl;
    } else {
        std::cout << " Директория Results уже существует" << std::endl;
    }

    std::string log_path = results_dir + "/2_log.txt";
    std::ofstream log_file(log_path);
    
    if (!log_file.is_open()) {
        std::cout << " Ошибка: Не удалось открыть файл для записи!" << std::endl;
        return 1;
    }
    
    std::cout << " Файл для записи результатов открыт: " << log_path << std::endl;

    const int num_tests = 3;
    double base_time = 0.0;
    


    for (size_t size : sizes) {
        std::cout << "\n🔧 Обрабатываем векторы размером: " << size << std::endl;
        log_file << "Vector size: " << size << "\n";

        std::cout << "    Генерируем случайные данные для двух векторов..." << std::endl;
        std::vector<int> a(size), b(size);
        for (size_t i = 0; i < size; ++i) {
            a[i] = dist(gen);
            b[i] = dist(gen);
        }
        std::cout << "    Данные сгенерированы" << std::endl;

        std::cout << "     Выполняем базовый замер (1 поток)..." << std::endl;
        {
            double total_time = 0.0;
            for (int t = 0; t < num_tests; ++t) {
                auto start = std::chrono::high_resolution_clock::now();
                scalar_production(a, b, 1);
                auto end = std::chrono::high_resolution_clock::now();
                total_time += std::chrono::duration<double, std::milli>(end - start).count();
            }
            base_time = total_time / num_tests;
            double speedup = 1.0;
            double efficiency = 1.0;
            log_file << "Threads: 1\n";
            log_file << "  Time: " << base_time << " ms (speedup: " << speedup << "x, efficiency: " << efficiency << ")\n";
        }
        std::cout << "    Базовый замер завершен: " << base_time << " мс" << std::endl;

        std::cout << "   Начинаем тестирование с разным количеством потоков..." << std::endl;
        for (int threads : thread_counts) {
            if (threads == 1) continue;

            std::cout << " Тестируем " << threads << " потоков..." << std::endl;

            double total_time = 0.0;
            for (int t = 0; t < num_tests; ++t) {
                auto start = std::chrono::high_resolution_clock::now();
                scalar_production(a, b, threads);
                auto end = std::chrono::high_resolution_clock::now();
                total_time += std::chrono::duration<double, std::milli>(end - start).count();
            }
            double avg_time = total_time / num_tests;
            double speedup = base_time / avg_time;
            double efficiency = speedup / threads;

            log_file << "Threads: " << threads << "\n";
            log_file << "  Time: " << avg_time << " ms (speedup: " << speedup << "x, efficiency: " << efficiency << ")\n";
            
            std::cout << " " << threads << " потоков: "
                      << avg_time << " мс (ускорение: " << speedup << "x)" << std::endl;
        }
        log_file << "--------------------------------------\n";
        std::cout << " Векторы размером " << size << " полностью обработаны" << std::endl;
    }

    log_file.close();
    std::cout << " Результаты сохранены в файл: " << log_path << std::endl;
    std::cout << " Программа завершена успешно!" << std::endl;
    
    return 0;
}

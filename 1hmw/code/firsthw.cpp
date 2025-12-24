#include <iostream>
#include <vector>
#include <omp.h>
#include <limits>
#include <chrono>
#include <fstream>
#include <random>
#include <sys/stat.h>

void no_reduction_method(const std::vector<int>& vec, int num_threads) {
    omp_set_num_threads(num_threads);
    int n = static_cast<int>(vec.size());

    int max_val = std::numeric_limits<int>::min();
    int min_val = std::numeric_limits<int>::max();

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        #pragma omp critical
        {
            if (vec[i] > max_val) max_val = vec[i];
            if (vec[i] < min_val) min_val = vec[i];
        }
    }
}

void reduction_method(const std::vector<int>& vec, int num_threads) {
    omp_set_num_threads(num_threads);
    int n = static_cast<int>(vec.size());

    int max_val = std::numeric_limits<int>::min();
    int min_val = std::numeric_limits<int>::max();

    #pragma omp parallel for reduction(max: max_val) reduction(min: min_val)
    for (int i = 0; i < n; i++) {
        if (vec[i] > max_val) max_val = vec[i];
        if (vec[i] < min_val) min_val = vec[i];
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
    std::cout << "🔄 Начинаем выполнение программы..." << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 10000);

    std::vector<int> thread_counts = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<size_t> sizes = { 100000, 500000, 1000000, 5000000 };

    std::string results_dir = "./Results";
    
    std::cout << " Проверяем наличие директории Results..." << std::endl;
    if (!directory_exists(results_dir)) {
        std::cout << " Создаем директорию Results..." << std::endl;
        if (!create_directory(results_dir)) {
            std::cout << " Ошибка: Не удалось создать директорию Results!" << std::endl;
            return 1;
        }
        std::cout << "Директория Results создана успешно" << std::endl;
    } else {
        std::cout << " Директория Results уже существует" << std::endl;
    }

    std::string log_path = results_dir + "/1_log.txt";
    std::ofstream log_file(log_path);
    
    if (!log_file.is_open()) {
        std::cout << " Ошибка: Не удалось открыть файл для записи!" << std::endl;
        return 1;
    }
    
    std::cout << " Файл для записи результатов открыт: " << log_path << std::endl;

   

    const int num_tests = 5;

    for (size_t size : sizes) {
        std::cout << "\n🔧 Обрабатываем вектор размером: " << size << std::endl;
        log_file << "Vector size: " << size << "\n";

        std::cout << "    Генерируем случайные данные..." << std::endl;
        std::vector<int> vec(size);
        for (size_t i = 0; i < size; i++) {
            vec[i] = dist(gen);
        }
        std::cout << "    Данные сгенерированы" << std::endl;

        std::cout << "    Выполняем базовые замеры (без reduction, 1 поток)..." << std::endl;
        double base_time_no_red = 0.0;
        {
            double time_one_thread = 0.0;
            for (int test = 0; test < num_tests; test++) {
                auto start = std::chrono::high_resolution_clock::now();
                no_reduction_method(vec, 1);
                auto end = std::chrono::high_resolution_clock::now();
                time_one_thread += std::chrono::duration<double, std::milli>(end - start).count();
            }
            base_time_no_red = time_one_thread / num_tests;

            double speedup_one_no_red = (base_time_no_red > 0) ? base_time_no_red / base_time_no_red : 1.0;
            double efficiency_one_no_red = speedup_one_no_red / 1.0;

            log_file << "Threads: 1\n";
            log_file << "  No reduction: " << base_time_no_red << " ms " << "(speedup: " << speedup_one_no_red << "x, efficiency: " << efficiency_one_no_red << ")\n";
        }
        std::cout << "   Базовые замеры (без reduction) завершены" << std::endl;

        std::cout << "  Выполняем базовые замеры (с reduction, 1 поток)..." << std::endl;
        double base_time_red = 0.0;
        {
            double time_one_thread_red = 0.0;
            for (int test = 0; test < num_tests; test++) {
                auto start = std::chrono::high_resolution_clock::now();
                reduction_method(vec, 1);
                auto end = std::chrono::high_resolution_clock::now();
                time_one_thread_red += std::chrono::duration<double, std::milli>(end - start).count();
            }
            base_time_red = time_one_thread_red / num_tests;

            double speedup_one_red = (base_time_red > 0) ? base_time_red / base_time_red : 1.0;
            double efficiency_one_red = speedup_one_red / 1.0;

            log_file << "  Reduction: " << base_time_red << " ms " << "(speedup: " << speedup_one_red << "x, efficiency: " << efficiency_one_red << ")\n";
        }
        std::cout << "   Базовые замеры (с reduction) завершены" << std::endl;

        std::cout << "   Начинаем тестирование с разным количеством потоков..." << std::endl;
        for (int threads : thread_counts) {
            if (threads == 1) continue;

            std::cout << " Тестируем " << threads << " потоков..." << std::endl;

            double no_reduction_time = 0.0;
            for (int test = 0; test < num_tests; test++) {
                auto start = std::chrono::high_resolution_clock::now();
                no_reduction_method(vec, threads);
                auto end = std::chrono::high_resolution_clock::now();
                no_reduction_time += std::chrono::duration<double, std::milli>(end - start).count();
            }
            no_reduction_time /= num_tests;
            double speedup_no_red = (base_time_no_red > 0) ? base_time_no_red / no_reduction_time : 0.0;
            double efficiency_no_red = speedup_no_red / threads;

            log_file << "Threads: " << threads << "\n";
            log_file << " No reduction: " << no_reduction_time << " ms " << "(speedup: " << speedup_no_red << "x, efficiency: " << efficiency_no_red << ")\n";

            // Тест с reduction
            double reduction_time = 0.0;
            for (int test = 0; test < num_tests; test++) {
                auto start = std::chrono::high_resolution_clock::now();
                reduction_method(vec, threads);
                auto end = std::chrono::high_resolution_clock::now();
                reduction_time += std::chrono::duration<double, std::milli>(end - start).count();
            }
            reduction_time /= num_tests;
            double speedup_red = (base_time_red > 0) ? base_time_red / reduction_time : 0.0;
            double efficiency_red = speedup_red / threads;

            log_file << " Reduction: " << reduction_time << " ms " << "(speedup: " << speedup_red << "x, efficiency: " << efficiency_red << ")\n";
            
            std::cout <<  threads << " потоков протестированы" << std::endl;
        }
        std::cout << "Размер вектора " << size << " полностью обработан" << std::endl;
    }

    log_file.close();
    std::cout << " Результаты сохранены в файл: " << log_path << std::endl;
    std::cout << " Программа завершена успешно!" << std::endl;
    
    return 0;
}

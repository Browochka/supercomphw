#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <cmath>
#include <sys/stat.h>
#include <unistd.h>

double f(double x) {
    return std::sin(x);
}

void compute_integral(double a, double b, double N, int num_threads, double& result) {
    omp_set_num_threads(num_threads);

    const double h = (b - a) / N;
    double local_sum = 0.0;

    for (long long i = 0; i < static_cast<long long>(N); ++i) {
        double x_i = a + (i + 0.5) * h;
        local_sum += f(x_i);
    }

    result = local_sum * h;
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
    std::cout << " Начинаем вычисление интеграла методом прямоугольников..." << std::endl;
    
    int max_procs = get_available_processors();
    std::cout << "💻 Доступно процессоров: " << max_procs << std::endl;
    
    const double a = 0.0;
    const std::vector<double> b_values = { 1000, 10000, 100000, 1000000, 10000000, 50000000 };

    std::vector<int> thread_counts_all = { 1, 2, 4, 6, 8, 12};
    std::vector<int> thread_counts;
    
    for (int t : thread_counts_all) {
        if (t <= max_procs * 2) {
            thread_counts.push_back(t);
        }
    }
    
    std::cout << "🧵 Тестируемые количества потоков: ";
    for (int t : thread_counts) std::cout << t << " ";
    std::cout << std::endl;

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
        std::cout << " Директория Results уже существует" << std::endl;
    }

    std::string log_path = results_dir + "/3_log.txt";
    std::ofstream log_file(log_path);
    
    if (!log_file.is_open()) {
        std::cerr << " Ошибка: Не удалось открыть файл для записи!\n";
        return 1;
    }
    
    std::cout << " Файл для записи результатов открыт: " << log_path << std::endl;

    const int num_tests = 3;
    double base_time = 0.0;



    for (double b : b_values) {
        const double N = b;
        const double h = (b - a) / N;

        std::cout << "\n🔧 Вычисляем интеграл на интервале [0, " << b << "]" << std::endl;
        std::cout << "   📊 Параметры: N = " << N << ", шаг h = " << h << std::endl;
        
        log_file << "Interval: [" << a << ", " << b << "], N = " << N << ", h = " << h << "\n";

        {
            std::cout << "   ⏱️  Выполняем базовый замер (1 поток)..." << std::endl;
            double total = 0.0;
            for (int t = 0; t < num_tests; ++t) {
                double res;
                const auto start = std::chrono::high_resolution_clock::now();
                compute_integral(a, b, N, 1, res);
                const auto end = std::chrono::high_resolution_clock::now();
                total += std::chrono::duration<double, std::milli>(end - start).count();
            }
            base_time = total / num_tests;
            log_file << "Threads: 1\n";
            log_file << "  Time: " << base_time << " ms (speedup: 1x, efficiency: 1)\n";
            std::cout << "    Базовый замер завершен: " << base_time << " мс" << std::endl;
        }

        std::cout << "   Начинаем тестирование с разным количеством потоков..." << std::endl;
        for (int threads : thread_counts) {
            if (threads == 1) continue;

            std::cout << "  Тестируем " << threads << " потоков..." << std::endl;

            double total = 0.0;
            for (int t = 0; t < num_tests; ++t) {
                double res;
                const auto start = std::chrono::high_resolution_clock::now();
                compute_integral(a, b, N, threads, res);
                const auto end = std::chrono::high_resolution_clock::now();
                total += std::chrono::duration<double, std::milli>(end - start).count();
            }
            const double avg_time = total / num_tests;
            const double speedup = base_time / avg_time;
            const double efficiency = speedup / threads;

            log_file << "Threads: " << threads << "\n";
            log_file << "  Time: " << avg_time << " ms (speedup: " << speedup << "x, efficiency: " << efficiency << ")\n";
            
            std::cout << " " << threads << " потоков: "
                      << avg_time << " мс (ускорение: " << speedup << "x)" << std::endl;
        }
        log_file << "--------------------------------------\n";
        std::cout << " Интервал [0, " << b << "] полностью обработан" << std::endl;
    }

    log_file.close();
    std::cout << " Результаты сохранены в файл: " << log_path << std::endl;
    std::cout << " Программа завершена успешно!" << std::endl;
    
    return 0;
}

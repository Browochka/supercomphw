#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <sys/stat.h>

bool directory_exists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

bool create_directory(const std::string& path) {
    return mkdir(path.c_str(), 0755) == 0;
}

void test_sections(int N, int D, const std::string& filename, int num_threads) {
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);

    std::vector<std::vector<double>> buffer;
    buffer.reserve(N);

    std::mutex mtx;
    std::atomic<bool> finished{false};
    std::atomic<bool> file_error{false};

    double scal = 0.0;

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            std::ifstream ifs(filename);
            if (!ifs.is_open()) {
                std::cerr << "Ошибка: не удалось открыть файл " << filename << std::endl;
                file_error = true;
                finished = true;
            } else {
                int total_vectors;
                int vector_dim;
                ifs >> total_vectors >> vector_dim;
                
                if (vector_dim != D) {
                    std::cerr << "Ошибка: размерность векторов в файле (" << vector_dim
                              << ") не соответствует ожидаемой (" << D << ")" << std::endl;
                    file_error = true;
                    finished = true;
                } else if (total_vectors < N) {
                    std::cerr << "Ошибка: в файле только " << total_vectors
                              << " векторов, а требуется " << N << std::endl;
                    file_error = true;
                    finished = true;
                } else {
                    for (int i = 0; i < N; ++i) {
                        std::vector<double> vec(D);
                        for (int j = 0; j < D; ++j) {
                            ifs >> vec[j];
                        }
                        {
                            std::lock_guard<std::mutex> lock(mtx);
                            buffer.push_back(std::move(vec));
                        }
                    }
                    finished = true;
                }
            }
        }

        #pragma omp section
        {
            while (!finished || !buffer.empty()) {
                if (file_error) {
                    break;
                }
                
                std::vector<std::vector<double>> local_copy;

                {
                    std::lock_guard<std::mutex> lock(mtx);
                    if (!buffer.empty()) {
                        local_copy.swap(buffer);
                    }
                }

                if (!local_copy.empty()) {
                    double local_scal = 0.0;
                    #pragma omp parallel for reduction(+:local_scal)
                    for (int i = 0; i < (int)local_copy.size() - 1; ++i) {
                        double sum = 0.0;
                        for (int k = 0; k < D; ++k) {
                            sum += local_copy[i][k] * local_copy[i + 1][k];
                        }
                        local_scal += sum;
                    }
                    scal += local_scal;
                }
                else if (!finished) {
                    std::this_thread::yield();
                }
            }
        }
    }
    
    if (file_error) {
        std::cerr << "Функция test_sections завершена с ошибкой" << std::endl;
    }
}

int main() {
    std::cout << "Начинаем выполнение программы test_sections..." << std::endl;
    
    std::vector<int> thread_counts_all = {1, 2, 4, 6, 8, 12, 16};
    std::vector<int> thread_counts;
    for (int t : thread_counts_all) {
        if (t <= 12) {
            thread_counts.push_back(t);
        }
    }
    
    // Изменяем пары размеров под ваши файлы
    std::vector<std::pair<int, int>> size_pairs = {
        {500, 100},   // vectors_500_100.txt
        {1000, 50},   // vectors_1000_50.txt
        {5000, 50},   // vectors_5000_50.txt
        {1000, 1000}  // vectors_1000_1000.txt
    };

    std::string results_dir = "./Results";
    std::cout << "Проверяем наличие директории Results..." << std::endl;
    if (!directory_exists(results_dir)) {
        std::cout << "Создаем директорию Results..." << std::endl;
        if (!create_directory(results_dir)) {
            std::cerr << "Ошибка: не удалось создать директорию Results!" << std::endl;
            return 1;
        }
        std::cout << "Директория Results создана" << std::endl;
    } else {
        std::cout << "Директория Results уже существует" << std::endl;
    }

    std::string log_path = results_dir + "/8_log.txt";
    std::ofstream log_file(log_path);
    if (!log_file.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл для записи!" << std::endl;
        return 1;
    }
    
    std::cout << "Файл для записи результатов открыт: " << log_path << std::endl;

    const int num_tests = 3;

    for (auto& p : size_pairs) {
        int N = p.first;
        int D = p.second;
        // Используем ваши имена файлов
        std::string filename = "vectors_" + std::to_string(N) + "_" + std::to_string(D) + ".txt";
        
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "Тестируем: N=" << N << ", D=" << D << ", файл=" << filename << std::endl;
        
        std::ifstream test_file(filename);
        if (!test_file.is_open()) {
            std::cerr << "Ошибка: файл " << filename << " не найден." << std::endl;
            std::cerr << "Убедитесь, что файл находится в той же директории, что и программа." << std::endl;
            log_file << "Size: " << N << " vectors of dimension " << D << " from file " << filename << " (FILE NOT FOUND)\n";
            log_file << "--------------------------------------\n";
            continue;
        }
        
        // Проверяем заголовок файла
        int file_N, file_D;
        test_file >> file_N >> file_D;
        test_file.close();
        
        if (file_N < N) {
            std::cerr << "Ошибка: в файле " << filename << " заявлено " << file_N
                      << " векторов, а требуется " << N << std::endl;
            log_file << "Size: " << N << " vectors of dimension " << D << " from file " << filename
                     << " (INSUFFICIENT DATA: " << file_N << " vectors declared)\n";
            log_file << "--------------------------------------\n";
            continue;
        }
        
        if (file_D != D) {
            std::cerr << "Ошибка: в файле " << filename << " размерность " << file_D
                      << ", а ожидается " << D << std::endl;
            log_file << "Size: " << N << " vectors of dimension " << D << " from file " << filename
                     << " (DIMENSION MISMATCH: " << file_D << " vs " << D << ")\n";
            log_file << "--------------------------------------\n";
            continue;
        }
        
        log_file << "Size: " << N << " vectors of dimension " << D << " from file " << filename << "\n";

        std::cout << "Выполняем базовый тест (1 поток)..." << std::endl;
        double base_time = 0.0;
        double total_time = 0.0;
        for (int t = 0; t < num_tests; ++t) {
            auto start = std::chrono::high_resolution_clock::now();
            test_sections(N, D, filename, 1);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        base_time = total_time / num_tests;
        log_file << "Threads: 1\n";
        log_file << " Time: " << base_time << " ms (speedup: 1.0x, efficiency: 1.0)\n";
        std::cout << "Базовый тест завершен: " << base_time << " мс" << std::endl;

        for (int threads : thread_counts) {
            if (threads == 1) continue;
            
            std::cout << "Тестируем с " << threads << " потоками..." << std::endl;
            total_time = 0.0;
            for (int t = 0; t < num_tests; ++t) {
                auto start = std::chrono::high_resolution_clock::now();
                test_sections(N, D, filename, threads);
                auto end = std::chrono::high_resolution_clock::now();
                total_time += std::chrono::duration<double, std::milli>(end - start).count();
            }
            double avg_time = total_time / num_tests;
            double speedup = base_time / avg_time;
            double efficiency = speedup / threads;
            
            log_file << "Threads: " << threads << "\n";
            log_file << " Time: " << avg_time << " ms (speedup: " << speedup << "x, efficiency: " << efficiency << ")\n";
            
            std::cout << "Тест с " << threads << " потоками завершен: " << avg_time << " мс (ускорение: " << speedup << "x)" << std::endl;
        }
        log_file << "--------------------------------------\n";
        std::cout << "Тестирование для N=" << N << ", D=" << D << " завершено" << std::endl;
    }

    log_file.close();
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Результаты сохранены в файл: " << log_path << std::endl;
    std::cout << "Программа завершена успешно" << std::endl;
    
    return 0;
}

#include <iostream>
#include <fcntl.h>
#include <assert.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>
#include <iomanip>

// Wrapper around a pointer, for reading values from byte sequence.
class Reader {
    public:
        Reader(const char *p) : ptr{p} {}
        template <typename T>
        Reader &operator>>(T &o) {
            // Assert alignment.
            assert(uintptr_t(ptr)%sizeof(T) == 0);
            o = *(T *) ptr;
            ptr += sizeof(T);
            return *this;
        }
    private:
        const char *ptr;
};

void dump(const std::string &fn);

int
main(int argc, char **argv) {

    for (int i = 1; i < argc; i++) {
        dump(argv[i]);
    }
}

void
dump(const std::string &fn) {

    std::cout << fn << std::endl;

    /*
     * Use mmap() for convenience.
     */

    int fd = open(fn.c_str(), O_RDONLY);
    if (fd < 0) {
        int en = errno;
        std::cerr << "Couldn't open " << fn << ": " << strerror(en) << "." << std::endl;
        exit(2);
    }

    // Get the actual size of the file.
    struct stat sb;
    int rv = fstat(fd, &sb); assert(rv == 0);
    // std::cout << sb.st_size << std::endl;

    // Use some flags that will hopefully improve performance.
    void *vp = mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (vp == MAP_FAILED) {
        int en = errno;
        fprintf(stderr, "mmap() failed: %s\n", strerror(en));
        exit(3);
    }
    char *file_mem = (char *) vp;

    // Tell the kernel that it should evict the pages as soon as possible.
    rv = madvise(vp, sb.st_size, MADV_SEQUENTIAL|MADV_WILLNEED); assert(rv == 0);

    rv = close(fd); assert(rv == 0);

    // Prefix to print before every line, to improve readability.
    std::string pref("    ");

    /*
     * Read file type string.
     */
    int n = strnlen(file_mem, 8);
    std::string file_type(file_mem, n);
    std::cout << pref << "File type string: " << file_type << std::endl;

    // Start to read data, skip the file type string.
    Reader reader{file_mem + 8};

    // TODO: Code below is repetitive, cleanup.
    // TODO: Add io manip to print with commas.

    if (file_type == "TRAINING") {

        uint64_t id;
        uint64_t n_points;
        uint64_t n_dims;

        reader >> id >> n_points >> n_dims;

        std::cout << pref << "Training file ID: " << std::hex << std::setw(16) << std::setfill('0') << id << std::dec << std::endl;
        std::cout << pref << "Number of points: " << n_points << std::endl;
        std::cout << pref << "Number of dimensions: " << n_dims << std::endl;
        for (std::uint64_t i = 0; i < n_points; i++) {
            std::cout << pref << "Point " << i << ": ";
            for (std::uint64_t j = 0; j < n_dims; j++) {
                float f;
                reader >> f;
                std::cout << std::fixed << std::setprecision(6) << std::setw(15) << std::setfill(' ') << f;
                // Add comma.
                if (j < n_dims - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << std::endl;
        }

    } else if (file_type == "QUERY") {

        uint64_t id;
        uint64_t n_queries;
        uint64_t n_dims;
        uint64_t n_neighbors;

        reader >> id >> n_queries >> n_dims >> n_neighbors;

        std::cout << pref << "Query file ID: " << std::hex << std::setw(16) << std::setfill('0') << id << std::dec << std::endl;
        std::cout << pref << "Number of queries: " << n_queries << std::endl;
        std::cout << pref << "Number of dimensions: " << n_dims << std::endl;
        std::cout << pref << "Number of neighbors to return for each point: " << n_neighbors << std::endl;
        for (std::uint64_t i = 0; i < n_queries; i++) {
            std::cout << pref << "Query " << i << ": ";
            for (std::uint64_t j = 0; j < n_dims; j++) {
                float f;
                reader >> f;
                std::cout << std::fixed << std::setprecision(6) << std::setw(15) << std::setfill(' ') << f;
                // Add comma.
                if (j < n_dims - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << std::endl;
        }

    } else if (file_type == "RESULT") {

        uint64_t training_id;
        uint64_t query_id;
        uint64_t result_id;
        uint64_t n_queries;
        uint64_t n_dims;
        uint64_t n_neighbors;

        reader >> training_id >> query_id >> result_id >> n_queries >> n_dims >> n_neighbors;

        std::cout << pref << "Training file ID: " << std::hex << std::setw(16) << std::setfill('0') << training_id << std::dec << std::endl;
        std::cout << pref << "Query file ID: " << std::hex << std::setw(16) << std::setfill('0') << query_id << std::dec << std::endl;
        std::cout << pref << "Result file ID: " << std::hex << std::setw(16) << std::setfill('0') << result_id << std::dec << std::endl;
        std::cout << pref << "Number of queries: " << n_queries << std::endl;
        std::cout << pref << "Number of dimensions: " << n_dims << std::endl;
        std::cout << pref << "Number of neighbors returned for each query: " << n_neighbors << std::endl;
        for (std::uint64_t i = 0; i < n_queries; i++) {
            std::cout << pref << "Result " << i << ": ";
            for (std::uint64_t j = 0; j < n_dims; j++) {
                float f;
                reader >> f;
                std::cout << std::fixed << std::setprecision(6) << std::setw(15) << std::setfill(' ') << f;
                // Add comma.
                if (j < n_dims - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << std::endl;
        }

    } else {
        std::cerr << "Unknown file type: " << file_type << std::endl;
        exit(2);
    }

    rv = munmap(file_mem, sb.st_size); assert(rv == 0);
}

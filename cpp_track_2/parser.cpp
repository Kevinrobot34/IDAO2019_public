#include "./parser.h"

inline bool not_number(const char pen) {
    return !isdigit(pen) && !(pen == '.') && !(pen == '-');
}

void skip_to_number_pointer(const char *& pen) {
    while ((*pen) && not_number(*pen)) ++pen;
}

inline float square(const float x) {
    return x*x;
}

// https://stackoverflow.com/questions/5678932/fastest-way-to-read-numerical-values-from-text-file-in-c-double-in-this-case
template<class T>
T rip_uint_pointer(const char *&pen, T val = 0) {
    // Will return val if *pen is not a digit
    // WARNING: no overflow checks
    for (char c; (c = *pen ^ '0') <= 9; ++pen)
        val = val * 10 + c;
    return val;
}

template<class T>
T rip_float_pointer(const char *&pen) {
    static double const exp_lookup[]
        = {1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10,
           1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17};
    T sign = 1.;
    if (*pen == '-') {
        ++pen;
        sign = -1.;
    }
    uint64_t val = rip_uint_pointer<uint64_t>(pen);
    unsigned int neg_exp = 0;
    if (*pen == '.') {
        const char* const fracs = ++pen;
        val = rip_uint_pointer(pen, val);
        neg_exp  = pen - fracs;
    }
    return std::copysign(val*exp_lookup[neg_exp], sign);
}

// Warning: this is not a general-puropse parser, you have
// std::istream for that. As a rule, in the interest of speed, it
// doesn't check for input correctness and will have undefined
// behavior at incorrect input
class BufferedStream {
 public:
    explicit BufferedStream(std::istream& stream);
    // Discards data from the stream until ecountering a digit, "." or "-"
    void skip_to_number();
    // Reads a float from the stream, starting from the current character
    // and has undefined behaviour if there is no number at the
    // current position
    template<class T> T rip_float() {return rip_float_pointer<T>(pen);}
    // Reads an unsigned integer from stream, starting from the
    // current character and has undefined behaviour if there is no
    // number at the current position
    template<class T> T rip_uint() {return rip_uint_pointer<T>(pen);}
    // Reads a vector of floats of the given size from the stream,
    // skipping everything as needed
    template<class T>
    std::vector<T> fill_vector_float(const size_t size);
    // Reads a vector of unsigned ints of the given size from the stream,
    // skipping as needed. In principle, should be templated from
    // fill_vector_float, but if constexpr is C++17 :(
    template<class T>
    std::vector<T> fill_vector_uint(const size_t size);
    // Reads count floating point numbers and stores them into the
    // container pointed to by the iterator
    template<class IteratorType>
    void fill_iterator_float(const IteratorType& iterator, const size_t count);
    // Discards data from the stream until encountering the delimiter
    void skip_to_char(const char delimiter);
    // Discrads data from the stream until twice encountering the delimiter
    void skip_record(const char delimiter);

 private:
    void next_line();
    // Buffer size is measured to fit the longest line in the test dataset
    // but the code doesn't rely on it
    static const size_t BUFFER_SIZE = 1016;
    char buffer[BUFFER_SIZE];
    std::istream& stream;
    const char* pen;
};

void BufferedStream::next_line() {
    stream.getline(buffer, BUFFER_SIZE);
    pen = buffer;
}

BufferedStream::BufferedStream(std::istream& stream): stream(stream) {
    next_line();
}

void BufferedStream::skip_to_number() {
    skip_to_number_pointer(pen);
    while ((*pen) == 0) {
        next_line();
        skip_to_number_pointer(pen);
        // The skip stops either at 0-byte or
        // a number part
    }
}

template<class T>
std::vector<T> BufferedStream::fill_vector_float(const size_t size) {
    std::vector<T> result(size);
    fill_iterator_float<std::vector<float>::iterator>(result.begin(), size);
    return result;
}

template<class T>
std::vector<T> BufferedStream::fill_vector_uint(const size_t size) {
    std::vector<T> result(size);
    for (auto& value : result) {
        skip_to_number();
        value = rip_uint<T>();
    }
    return result;
}

void BufferedStream::skip_to_char(const char delimiter) {
    while ((*pen) != delimiter) {
        while ((*pen) && (*(++pen)) != delimiter) {}
        if (!(*pen)) next_line();
    }
}

void BufferedStream::skip_record(const char delimiter) {
    skip_to_char(delimiter);
    ++pen;
    skip_to_char(delimiter);
}

template<class IteratorType>
void BufferedStream::fill_iterator_float(const IteratorType& iterator, const size_t count) {
    for (IteratorType value = iterator; value != iterator + count; ++value) {
        skip_to_number();
        *value = rip_float<typename std::iterator_traits<IteratorType>::value_type>();
    }
}

void ugly_hardcoded_parse(std::istream& stream, size_t* id, std::vector<float>* result) {
    BufferedStream buffered_stream(stream);
    *id = buffered_stream.rip_uint<size_t>();
    buffered_stream.fill_iterator_float<std::vector<float>::iterator>(
        result->begin(), N_RAW_FEATURES - N_RAW_FEATURES_TAIL);
    // No need to skip, fill_vector takes care of it
    const size_t FOI_hits_N = (*result)[FOI_HITS_N_INDEX];
    const std::vector<float> FOI_hits_X = buffered_stream.fill_vector_float<float>(FOI_hits_N);
    const std::vector<float> FOI_hits_Y = buffered_stream.fill_vector_float<float>(FOI_hits_N);
    const std::vector<float> FOI_hits_Z = buffered_stream.fill_vector_float<float>(FOI_hits_N);
    const std::vector<float> FOI_hits_DX = buffered_stream.fill_vector_float<float>(FOI_hits_N);
    const std::vector<float> FOI_hits_DY = buffered_stream.fill_vector_float<float>(FOI_hits_N);
    buffered_stream.skip_record(DELIMITER);
    const std::vector<float> FOI_hits_T = buffered_stream.fill_vector_float<float>(FOI_hits_N);
    buffered_stream.skip_record(DELIMITER);
    const std::vector<size_t> FOI_hits_S = \
        buffered_stream.fill_vector_uint<size_t>(FOI_hits_N);

    std::array<size_t, N_STATIONS> closest_hit_per_station;
    std::array<float, N_STATIONS> closest_hit_distance;
    closest_hit_distance.fill(std::numeric_limits<float>::infinity());
    std::valarray<float> distance_x_sum(0.0f, N_STATIONS);
    std::valarray<float> distance_y_sum(0.0f, N_STATIONS);
    std::valarray<int> hit_count(0, N_STATIONS);
    std::valarray<float> first_vec(0.0f, 3);
    std::valarray<float> second_vec(0.0f, 3);
    std::valarray<float> third_vec(0.0f, 3);
    std::valarray<float> lextra_vec(0.0f, 3);
    std::valarray<float> origin_vec(0.0f, 3);
    float first_norm = 0.0f;
    float second_norm = 0.0f;
    float third_norm = 0.0f;
    float lextra_dot = 0.0f;
    float lextra_norm = 0.0f;
    float lextra_theta = 0.0f;
    float origin_dot = 0.0f;
    float origin_norm = 0.0f;
    float first_dot = 0.0f;
    float second_dot = 0.0f;
    float first_theta = 0.0f;
    float second_theta = 0.0f;
    const float RAD = 180 / 3.141592653589793;

    origin_vec[0] = (*result)[MATCHEDHIT_X_INDEX];
    origin_vec[1] = (*result)[MATCHEDHIT_X_INDEX+N_STATIONS];
    origin_vec[2] = (*result)[MATCHEDHIT_X_INDEX+N_STATIONS*2];
    origin_norm = std::sqrt((origin_vec * origin_vec).sum());

    for (size_t hit_index = 0; hit_index < FOI_hits_N; ++hit_index) {
        const size_t this_station = FOI_hits_S[hit_index];
        const float distance_x_2 = square(FOI_hits_X[hit_index] -
                                          (*result)[LEXTRA_X_INDEX + this_station]);
        const float distance_y_2 = square(FOI_hits_Y[hit_index] -
                                          (*result)[LEXTRA_Y_INDEX + this_station]);
        const float distance_2 = distance_x_2 + distance_y_2;
        //
        distance_x_sum[this_station] += distance_x_2;
        distance_y_sum[this_station] += distance_y_2;
        hit_count[this_station] += 1;
        //
        if (distance_2 < closest_hit_distance[this_station]) {
            closest_hit_distance[this_station] = distance_2;
            closest_hit_per_station[this_station] = hit_index;
            (*result)[FOI_FEATURES_START + this_station] = distance_x_2;
            (*result)[FOI_FEATURES_START + N_STATIONS + this_station] = distance_y_2;
        }
    }
    /* [closest_x_per_station, closest_y_per_station, closest_T_per_station,
       closest_z_per_station, closest_dx_per_station, closest_dy_per_station]) */
    for (size_t station = 0; station < N_STATIONS; ++station) {
        if (std::isinf(closest_hit_distance[station])) {
            for (size_t feature_index = 0; feature_index < FOI_FEATURES_PER_STATION; ++feature_index) {
                (*result)[FOI_FEATURES_START + feature_index * N_STATIONS + station] = EMPTY_FILLER;
            }
        }

        (*result)[AVERAGE_INDEX + station] = distance_x_sum[station] / hit_count[station];
        (*result)[AVERAGE_INDEX +  N_STATIONS + station] = distance_y_sum[station] / hit_count[station];
        (*result)[CLOSESTXY_INDEX + station] = (*result)[N_RAW_FEATURES + station] + (*result)[N_RAW_FEATURES + station + N_STATIONS];
        lextra_vec[0] = (*result)[LEXTRA_X_INDEX + station];
        lextra_vec[1] = (*result)[LEXTRA_X_INDEX + N_STATIONS + station];
        if (station == 0) {
          first_vec[0] = (*result)[MATCHEDHIT_X_INDEX + station + 1] - (*result)[MATCHEDHIT_X_INDEX + station];
          first_vec[1] = (*result)[MATCHEDHIT_X_INDEX + N_STATIONS + station + 1] - (*result)[MATCHEDHIT_X_INDEX + N_STATIONS + station];
          first_vec[2] = (*result)[MATCHEDHIT_X_INDEX + N_STATIONS * 2 + station + 1] - (*result)[MATCHEDHIT_X_INDEX + N_STATIONS *  2 + station];
          first_norm = std::sqrt((first_vec * first_vec).sum());
          lextra_vec[2] = 15261.423;
          lextra_dot = (first_vec * lextra_vec).sum();
          origin_dot  = (origin_vec * first_vec).sum();
          lextra_norm = std::sqrt((lextra_vec * lextra_vec).sum());
          lextra_theta = std::acos( lextra_dot / (first_norm * lextra_norm) ) * RAD;
          (*result)[MANGLEV2_INDEX + station] = lextra_theta;
          (*result)[MANGLE_INDEX + 2] = std::acos( origin_dot / (origin_norm * first_norm) ) * RAD;
        } else if (station == 1) {
          second_vec[0] = (*result)[MATCHEDHIT_X_INDEX + station + 1] - (*result)[MATCHEDHIT_X_INDEX + station];
          second_vec[1] = (*result)[MATCHEDHIT_X_INDEX + N_STATIONS + station + 1] - (*result)[MATCHEDHIT_X_INDEX + N_STATIONS + station];
          second_vec[2] = (*result)[MATCHEDHIT_X_INDEX + N_STATIONS * 2 + station + 1] - (*result)[MATCHEDHIT_X_INDEX + N_STATIONS *  2 + station];
          second_norm = std::sqrt((second_vec * second_vec).sum());
          lextra_vec[2] = 16467.137;
          lextra_dot = (second_vec * lextra_vec).sum();
          lextra_norm = std::sqrt((lextra_vec * lextra_vec).sum());
          lextra_theta = std::acos( lextra_dot / (second_norm * lextra_norm) ) * RAD;
          (*result)[MANGLEV2_INDEX + station] = lextra_theta;
        } else if (station == 2) {
          third_vec[0] = (*result)[MATCHEDHIT_X_INDEX + station + 1] - (*result)[MATCHEDHIT_X_INDEX + station];
          third_vec[1] = (*result)[MATCHEDHIT_X_INDEX + N_STATIONS + station + 1] - (*result)[MATCHEDHIT_X_INDEX + N_STATIONS + station];
          third_vec[2] = (*result)[MATCHEDHIT_X_INDEX + N_STATIONS * 2 + station + 1] - (*result)[MATCHEDHIT_X_INDEX + N_STATIONS *  2 + station];
          third_norm = std::sqrt((third_vec * third_vec).sum());
          lextra_vec[2] = 17660.324;
          lextra_dot = (third_vec * lextra_vec).sum();
          lextra_norm = std::sqrt((lextra_vec * lextra_vec).sum());
          lextra_theta = std::acos( lextra_dot / (third_norm * lextra_norm) ) * RAD;
          (*result)[MANGLEV2_INDEX + station] = lextra_theta;
        }
    }
    first_dot = (first_vec * second_vec).sum();
    second_dot = (third_vec * second_vec).sum();
    first_theta = std::acos( first_dot / (first_norm * second_norm) ) * RAD;
    second_theta = std::acos( second_dot / (second_norm * third_norm) ) * RAD;
    (*result)[MANGLE_INDEX] = first_theta;
    (*result)[MANGLE_INDEX + 1] = second_theta;


    buffered_stream.fill_iterator_float<std::vector<float>::iterator>(
        result->begin() + N_RAW_FEATURES - N_RAW_FEATURES_TAIL, N_RAW_FEATURES_TAIL);

    (*result)[PANGLE_INDEX] = (*result)[N_RAW_FEATURES - 2] / (*result)[N_RAW_FEATURES - 1];

}

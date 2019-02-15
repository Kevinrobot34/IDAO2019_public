// Copyright 2019, Nikita Kazeev, Higher School of Economics
#pragma once
#include <ctype.h>
#include <cstdint>
#include <cmath>

#include <iostream>
#include <vector>
#include <array>
#include <valarray>
#include <limits>

const size_t N_STATIONS = 4;
// const size_t FOI_FEATURES_PER_STATION = 6;
const size_t FOI_FEATURES_PER_STATION = 6;
// The structure of .csv is the following:
// id, <62 float features>, number of hits in FOI, <9 arrays of FOI hits features>, <2 float features>
const size_t N_RAW_FEATURES = 65;
const size_t N_RAW_FEATURES_TAIL = 2;
const size_t N_ANGLES = 3;
const size_t FOI_HITS_N_INDEX = 62;
const size_t LEXTRA_X_INDEX = 45;
const size_t LEXTRA_Y_INDEX = 49;
// const size_t N_FEATURES = N_RAW_FEATURES + N_STATIONS * FOI_FEATURES_PER_STATION + N_ANGLES;
const size_t N_FEATURES = 92;
const size_t FOI_FEATURES_START = N_RAW_FEATURES;
const float EMPTY_FILLER = 1000;

const size_t MATCHEDHIT_X_INDEX = 13;
const size_t FOI_ADDITIONAL_START = N_RAW_FEATURES + N_STATIONS * 6;

const size_t PANGLE_INDEX = 73;
const size_t AVERAGE_INDEX = 74;
const size_t MANGLE_INDEX = 82;
const size_t MANGLEV2_INDEX = 85;
const size_t CLOSESTXY_INDEX = 88;

const char DELIMITER = ',';

void ugly_hardcoded_parse(std::istream& stream, size_t* id, std::vector<float>* result);

#ifndef _TIMER_H_
#define _TIMER_H_
/**
 * \file Timer.h
 * \brief A timer class that provides a cross platform timer for use
 * in timing code progress with a high degree of accuracy.
 */
#ifdef _WIN32
/**
 * \typedef __int64 i64
 * \brief Maps the windows 64 bit integer to a uniform name
 */
typedef __int64 i64 ;
#else
/**
 * \typedef long long i64
 * \brief Maps the linux 64 bit integer to a uniform name
 */
typedef long long i64;
#include <sys/times.h>
#include <unistd.h>
#include <time.h>

#endif

#define SEQ_CAL_KEY_TIME 0
#define SEQ_INSERTION_TIME 1
#define PARALLEL_HASH_TIME 2
#define SEQ_HASH_TIME 3
#define STL_HASH_TIME 4

#define GET_KEY 5
#define SCATTER 6
#define GPU_SORT 7
#define UPDATE_ISA 8
#define R_SORT 9
#define NEIG_COM 10
#define PREFIX_SUM 11
#define SC_UNIQUE 12
#define B_SORT 13
#define SC_GROUP 14
#define UP_BLOCK 15
#define AS_BLOCK 16
#define D_TRANS 17
#define TOTAL 18

typedef struct CPerfCounterRec
{
    i64 _freq;
    i64 _clocks;
    i64 _start;
} CPerfCounter;

void Setup(int);

/**
 * \brief Start the timer
 * \sa Stop(), Reset(), Setup()
 */
void Start(int);

/**
 * \brief Stop the timer
 * \sa Start(), Reset(), Setup()
 */
void Stop(int);

/**
 * \brief Reset the timer to 0
 * \sa Start(), Stop(), Setup()
 */
void Reset(int);

/**
 * \return Amount of time that has accumulated between the \a Start()
 * and \a Stop() function calls
 */
double GetElapsedTime(int);

/**
 * Print the amount of time elapsed
 */

void PrintTime(int);

#endif 


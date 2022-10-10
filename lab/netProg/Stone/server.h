//
// Created by stone on 2022/10/9.
//

#ifndef NETPROG_SERVER_H
#define NETPROG_SERVER_H

#include <stdio.h>
#include <stdlib.h>

#include <netdb.h>
#include <netinet/in.h>
#include <unistd.h>
#include <time.h>
#include <ctime>
#include <pthread.h>

#include <string.h>

const int MAX_CLIENT_COUNT  = 10;
const int MAX_MESSAGE_LEN   = 10;
const int MAX_NICKNAME_LEN  = 10;
const int MAX_MESSAGE_SIZE  = 256;
const int MAX_NICKNAME_SIZE = 256;
const int MAX_BUFFER_SIZE   = 1024;
const int TIME_FORMAT_LEN   = 10;

typedef struct TIME {
    int hour;
    int min;
    int sec;
}my_time;

void get_time_format(char* time_format);
void check_op(int n);

struct tm* get_time();

int chars2int(char* message_len);
int stick_sending_msg(const char* msg, const char* nickname, char* dst);
int get_tens(int power);

static inline int reserve_socket_cell();

static inline void free_socket_cell(int cell);

static inline void notify_all(char* message, int message_size, char* nickname, int nickname_size, int skip);

static void* client_handler(void* arg);

#endif //NETPROG_SERVER_H

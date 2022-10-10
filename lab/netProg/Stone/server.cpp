#include "server.h"

int clients[MAX_CLIENT_COUNT]; // file handler
char is_active[MAX_CLIENT_COUNT];
int active_client_count;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

struct tm* get_time() {
    time_t ticks = time(NULL);
    struct tm* time = gmtime(&ticks);
    return time;
}

void get_time_format(char* time_format) {
    struct tm* time;
    time = get_time();
    int hour = time->tm_hour + 8;
    int min  = time->tm_min;
    int sec  = time->tm_sec;

    *(time_format) = '<';
    *(time_format + 1) = hour / 10 + 48;
    *(time_format + 2) = hour % 10 + 48;
    *(time_format + 3) = ':';
    *(time_format + 4) = min / 10 + 48;
    *(time_format + 5) = min % 10 + 48;
    *(time_format + 6) = ':';
    *(time_format + 7) = sec / 10 + 48;
    *(time_format + 8) = sec % 10 + 48;
    *(time_format + 9) = '>';
}

void check_op(int n) {
    if (n < 0) {
        perror("ERROR reading from socket");
        exit(1);
    }
}

int get_tens(int power) {
    int result = 1;
    for (int i = 0; i < power; i++) {
        result *= 10;
    }
    return result;
}

int stick_sending_msg(const char* msg, const char* nickname, char* dst) {
    char time_format[TIME_FORMAT_LEN];
    bzero(time_format, TIME_FORMAT_LEN);
    int msg_size = strlen(msg);
    int nickname_size = strlen(nickname);
    int length = 0;

    get_time_format(time_format);

    for (int i = 0; i < TIME_FORMAT_LEN; i++) {
        *(dst + i) = time_format[i];
    }
    length += TIME_FORMAT_LEN;
    *(dst + length) = ' ';
    length ++;
    *(dst + length) = '[';
    length ++;
    for (int i = 0; i < nickname_size; i++) {
        *(dst + length + i) = *(nickname + i);
    }
    length += nickname_size;
    *(dst + length) = ']';
    length ++;
    *(dst + length) = ' ';
    length ++;
    for (int i = 0; i < msg_size; i++) {
        *(dst + length + i) = *(msg + i);
    }
    length += msg_size;
    return length;
}

int chars2int(char* message_len) {
    int result = 0;
    for (int i = 0; i < MAX_MESSAGE_LEN; i++) {
        result += (int) (*(message_len + i) - 48) * get_tens(i);
    }
    return result;
}

static inline int reserve_socket_cell() {
    int cell = -1;
    for (int i = 0; i < MAX_CLIENT_COUNT; i++) {
        if (is_active[i] != 1) {
            cell = i;
            break;
        }
    }
    if (cell == -1) {
        printf("No available cell in is_active array.\n");
        for (int i = 0; i < MAX_CLIENT_COUNT; i++) {
            printf("is_active[%d] = %c\t", i, is_active[i]);
        }
    }
    pthread_mutex_lock(&mutex);
    is_active[cell] = 1;
    active_client_count ++;
    pthread_mutex_unlock(&mutex);
    return cell;
}

static inline void free_socket_cell(int cell) {
    pthread_mutex_lock(&mutex);
    is_active[cell] = 0;
    active_client_count --;
    pthread_mutex_unlock(&mutex);
}

static inline void notify_all(char* message, int message_size, char* nickname, int nickname_size, int skip) {
//    printf("msg_size: %d, nick_size: %d\n", message_size, nickname_size);
    for (int i = 0; i < MAX_CLIENT_COUNT; i++) {
        if (i == skip) {
            continue;
        }
        if (is_active[i] == 1) {
            char* output = (char*) calloc(512, sizeof(char));
            int n = stick_sending_msg(message, nickname, output);
//            printf("sticking msg length = %d\n", n);
            int rtn = write(clients[i], output, n);
            check_op(rtn);
        }
    }
}

static void* client_handler(void* arg) {
    int cell = (char*) arg - is_active;
    char message_len[MAX_MESSAGE_LEN], nickname_len[MAX_NICKNAME_LEN];
    char message[MAX_MESSAGE_SIZE], nickname[MAX_NICKNAME_SIZE];
    int message_size, nickname_size;
    int n;

    bzero(message, MAX_MESSAGE_SIZE);
    bzero(nickname, MAX_NICKNAME_SIZE);

    n = read(clients[cell], &nickname_len, MAX_NICKNAME_LEN);
    check_op(n);
    nickname_size = chars2int(nickname_len);

    n = read(clients[cell], &nickname, nickname_size);
    check_op(n);

    n = read(clients[cell], &message_len, MAX_MESSAGE_LEN);
    check_op(n);
    message_size = chars2int(message_len);

    n = read(clients[cell], &message, message_size);
    check_op(n);

    printf("Message get: %s\n", message);
    notify_all(message, message_size, nickname, nickname_size, cell);

    return NULL;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    int sockfd, newsockfd;
    uint16_t portno;
    unsigned int clilen;
//    char buffer[MAX_BUFFER_SIZE];
    struct sockaddr_in serv_addr, cli_addr;
//    ssize_t n;

    /* First call to socket() function */
    sockfd = socket(AF_INET, SOCK_STREAM, 0);

    if (sockfd < 0) {
        perror("ERROR opening socket");
        exit(1);
    }

    /* Initialize socket structure */

    /* Get portno */
    if (argc != 2) {
        fprintf(stderr, "usage: %s portno\n", argv[0]);
        exit(0);
    }
    portno = (uint16_t) atoi(argv[1]);

    bzero((char *)&serv_addr, sizeof(serv_addr));

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);

    /* Now bind the host address using bind() call.*/
    if (bind(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("ERROR on binding");
        close(sockfd);
        exit(1);
    }

    /* Now start listening for the clients, here process will
     * go in sleep mode and will wait for the incoming connection
     */

    listen(sockfd, 5);
    clilen = sizeof(cli_addr);

    /* Accept actual connection from the client */
    while (true) {
        newsockfd = accept(sockfd, (struct sockaddr *)&cli_addr, &clilen);
        if (active_client_count >= MAX_CLIENT_COUNT) {
            printf("Count overflow: %d/%d\n", active_client_count, MAX_CLIENT_COUNT);
            continue;
        }
        int cell = reserve_socket_cell();
        clients[cell] = newsockfd;
        pthread_t tid;
        if (pthread_create(&tid, NULL, client_handler, is_active + cell) != 0) {
            continue;
        }
        pthread_detach(tid);

//        if (newsockfd < 0) {
//            perror("ERROR on accept");
//            exit(1);
//        }
//
//        /* If connection is established then start communicating */
//        bzero(buffer, MAX_BUFFER_SIZE);
//        n = read(newsockfd, buffer, MAX_BUFFER_SIZE);
//
//        if (n < 0) {
//            perror("ERROR reading from socket");
//            exit(1);
//        }
//
//        printf("Here is the message: %s\n", buffer);
//
//        /* Write a response to the client */
//        n = write(newsockfd, "I got your message", 18);
//
//        if (n < 0) {
//            perror("ERROR writing to socket");
//            exit(1);
//        }
    }


    return 0;
}

#include <stdio.h>
#include <stdlib.h>

#include <netdb.h>
#include <netinet/in.h>
#include <unistd.h>
#include <iostream>

#include <string.h>

using namespace std;

const int NICK_SIZE = 256;
const int BODY_SIZE = 256;
const int DST_SIZE  = 512;

const int NICK_SIZE_LENGTH = 10;
const int BODY_SIZE_LENGTH = 10;

void attach_nickname(char* nickname, char* buffer, int nick_length) {
	for (int i = 0; i < nick_length; i++) {
		*(buffer + i) = *(nickname + i);
	}
}

bool judge_m(char* buffer) {
//	cout << strlen(buffer) / sizeof(char) << endl;
	//printf("length is %d, buffer is %s\n", strlen(buffer) / sizeof(char), buffer);
    if (strlen(buffer) / sizeof(char) != 2) {
        return false;
    } else {
        if (buffer[0] != 'm') {
            return false;
        } else {
            return true;
        }
    }
}

unsigned int get_tens(int power) {
	unsigned int result = 1;
	for (int i = 0; i < power; i++) {
		result *= 10;
	}
	return result;
}

void convert(unsigned int size, int* dst) {
	for (int i = NICK_SIZE_LENGTH - 1; i >= 0; i--) {
		*(dst + i) = size / get_tens(i);
//		printf("%d: %d\n", i, *(dst + i));
		size %= get_tens(i);
	}
}

void stick(char* dst, unsigned int nick_size, char* nickname, unsigned int body_size, char* body) {
//	printf("Got into sticking\n");
	int nick_size_digits[NICK_SIZE_LENGTH];
	int body_size_digits[BODY_SIZE_LENGTH];
	convert(nick_size, nick_size_digits);
	convert(body_size, body_size_digits);
//	printf("nick_size_digits[0] = %d\n", nick_size_digits[0]);
//	printf("body_size_digits[0] = %d\n", body_size_digits[0]);
//	printf("body = %s\n", body);
	for (unsigned int i = 0; i < NICK_SIZE_LENGTH; i++) {
		*(dst + i) = nick_size_digits[i] + 48;
	}
	for (unsigned int i = 0; i < nick_size; i++) {
		*(dst + NICK_SIZE_LENGTH + i) = *(nickname + i);
	}
	for (unsigned int i = 0; i < BODY_SIZE_LENGTH; i++) {
		*(dst + NICK_SIZE_LENGTH + nick_size + i) = body_size_digits[i] + 48;
	}
	for (unsigned int i = 0; i < body_size; i++) {
		*(dst + NICK_SIZE_LENGTH + nick_size + BODY_SIZE_LENGTH + i) = *(body + i);
//		printf("Fourth: %c\n", *(dst + NICK_SIZE_LENGTH + nick_size + BODY_SIZE_LENGTH))
	}
}

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  int sockfd, n;
  unsigned int nick_size, body_size;
  uint16_t portno;
  struct sockaddr_in serv_addr;
  struct hostent *server;

  char buffer[BODY_SIZE];
  char nickname[NICK_SIZE];
  char dst[DST_SIZE];
  
  memset(buffer, 0, BODY_SIZE);
  memset(nickname, 0, NICK_SIZE);
  memset(dst, 0, DST_SIZE);


  if (argc < 4) {
    fprintf(stderr, "usage %s hostname port\n", argv[0]);
    exit(0);
  }

  portno = (uint16_t)atoi(argv[2]);

  /* Create a socket point */
  sockfd = socket(AF_INET, SOCK_STREAM, 0);

  if (sockfd < 0) {
    perror("ERROR opening socket");
    exit(1);
  }

  server = gethostbyname(argv[1]);

  if (server == NULL) {
    fprintf(stderr, "ERROR, no such host\n");
    exit(0);
  }

  bzero((char *)&serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  bcopy(server->h_addr, (char *)&serv_addr.sin_addr.s_addr,
        (size_t)server->h_length);
  serv_addr.sin_port = htons(portno);

  /* Get nickname and its length */
  strcpy(nickname, argv[3]);
  nick_size = strlen(nickname);

  /* Now connect to the server */
  if (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
    perror("ERROR connecting");
    exit(1);
  }

  /* Now ask for a message from the user, this message
   * will be read by server
   */

  printf("Please enter the message after typing 'm': \n");
  bzero(buffer, 256);
//    printf("\nNickname: %s, length = %d\n", nickname, length);
//    attach_nickname(nickname, buffer, length);

    if (fgets(buffer, 255, stdin) == NULL) {
        perror("ERROR reading from stdin");
        exit(1);
    }

    while (!judge_m(buffer)) {
        printf("invalid input\n");
        if (fgets(buffer, 255, stdin) == NULL) {
            perror("ERROR reading from stdin");
            exit(1);
        }
    }

    if (fgets(buffer, 255, stdin) == NULL) {
        perror("ERROR reading from stdin");
        exit(1);
    }

//  printf("Buffer is %s\n", buffer);

  /* Get buffer length */
  body_size = strlen(buffer);

  /* Stick nickname and buffer contents to dst */
  stick(dst, nick_size, nickname, body_size, buffer);

  /* Send message to the server */
  n = write(sockfd, dst, strlen(dst));

  if (n < 0) {
    perror("ERROR writing to socket");
    exit(1);
  }

  /* Now read server response */
  bzero(buffer, 256);
  n = read(sockfd, buffer, 255);

  if (n < 0) {
    perror("ERROR reading from socket");
    exit(1);
  }

  printf("%s\n", buffer);
  return 0;
}

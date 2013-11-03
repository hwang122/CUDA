#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_ERR(x)                                            \
    if (x != cudaSuccess) {                                     \
        fprintf(stderr, "%s in %s at line %d\n",                \
                cudaGetErrorString(err), __FILE__, __LINE__);   \
        exit(-1);                                               \
    }                                                           \

#define FILE_LINE 1024000
#define LINE_WIDTH 256

__device__ char *d_strstr(const char *str1, const char *str2, int width){
    char *cp = (char *)str1;
    char *s1, *s2;

    if(!*str2)
        return ((char*)str1);

    int i = 0;
    while(i < width)
    {
        s1 = cp;
        s2 = (char *)str2;

        while(*s1 && *s2 && !(*s1 - *s2))
            s1++, s2++;

        if(!*s2)
            return cp;
        cp++;
        i++;
    }

    return NULL;
}

__device__ char *d_memcpy(char *dest, char *src, int count)
{
    char *result = dest;

    if(dest <= src || dest >= (src + count))
    {
        while(count--)
            *(char *)dest++ = *(char *)src++;
    }
    else
    {
        dest += count - 1;
        src += count - 1;

        while(count--)
            *(char *)dest-- = *(char *)src--;
    }

    return result;
}

__global__ void d_Grep(char *d_File, char *d_regex, char *result, int line, int width){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    char *pch;
    if(i < line)
    {
        pch = d_strstr(&d_File[i*width], d_regex, width);
        if(pch != NULL)
            d_memcpy(&result[i*width], &d_File[i*width], sizeof(char)*width);
    }
}

int main(int argc, char* argv[])
{
    cudaError_t err;

    char *Filename = argv[1];
	char *Regexp = argv[2];
    if(Regexp==NULL||Filename==NULL){
        printf("Usage: #./program [file name] [searching words]");
        return -1;
    }
    FILE *f;
    f = fopen(Filename, "r");
    if(f == NULL)
    {
        printf("Fail to open file!\n");
        return -1;
    }

    char *file;
    char *result;
    int i;

    file = (char *)malloc(sizeof(char)*FILE_LINE*LINE_WIDTH);
    result = (char *)malloc(sizeof(char)*FILE_LINE*LINE_WIDTH);

    fgets(file, FILE_LINE*LINE_WIDTH, f);

    char *d_file, *d_regex, *d_result;
    err = cudaMalloc((void**) &d_file, sizeof(char)*FILE_LINE*LINE_WIDTH);
    CHECK_ERR(err);

    err = cudaMalloc((void**) &d_regex, strlen(Regexp));
    CHECK_ERR(err);

    err = cudaMalloc((void**) &d_result, sizeof(char)*FILE_LINE*LINE_WIDTH);
    CHECK_ERR(err);

    err = cudaMemcpy(d_file, file, sizeof(char)*FILE_LINE*LINE_WIDTH, cudaMemcpyHostToDevice);
    CHECK_ERR(err);

    err = cudaMemcpy(d_regex, Regexp,  strlen(Regexp), cudaMemcpyHostToDevice);
    CHECK_ERR(err);

    int numThread = 256;
    int numBlock = ceil((double)FILE_LINE/numThread);
    d_Grep<<<numBlock, numThread>>>(d_file, d_regex, d_result, FILE_LINE, LINE_WIDTH);

    err = cudaMemcpy(result, d_result, sizeof(char)*FILE_LINE*LINE_WIDTH, cudaMemcpyDeviceToHost);
    CHECK_ERR(err);

    for(i = 0; i < FILE_LINE; i++)
    {
        if(&result[i*LINE_WIDTH] != NULL)
            printf("%s", &result[i*LINE_WIDTH]);
    }

    return 0;
}

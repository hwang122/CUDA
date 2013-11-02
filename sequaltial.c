#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FILE_LINE 300
#define LINE_WIDTH 256

char *my_strstr(const char *str1, const char *str2)
{
    char *cp = (char *)str1;
    char *s1, *s2;

    if(!*str2)
        return ((char*)str1);

    int i = 0;
    while(i < LINE_WIDTH)
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

char *my_memcpy(char *dest, char *src, int count)
{
    char *result = dest;

    if(dest <= src || dest >= (src + count))
    {
        while(count--)
        {
            *(char *)dest++ = *(char *)src++;
        }
    }
    else
    {
        dest += count - 1;
        src += count - 1;

        while(count--)
        {
            *(char *)dest-- = *(char *)src--;
        }
    }

    return result;
}

int main()
{
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

    char **file;
    int i;

    file = (char **)malloc(sizeof(char*)*FILE_LINE);
    //keep the continuity of memory
    file[0] = (char *)malloc(sizeof(char)*FILE_LINE*LINE_WIDTH);
    for(i = 1; i < FILE_LINE; i++)
        file[i] = file[i-1] + LINE_WIDTH;

    for(i = 0; i < FILE_LINE; i++)
    {
        fgets(file[i], LINE_WIDTH, f);
    }

    char *d_file;

    d_file = (char *)malloc(sizeof(char)*FILE_LINE*LINE_WIDTH);

    memcpy(d_file, &file[0][0], sizeof(char) * FILE_LINE * LINE_WIDTH);

    int offset = 0;
    char *pch;
    char *result = (char *)malloc(sizeof(char)*FILE_LINE*LINE_WIDTH);
    for(i = 0; i < FILE_LINE; i++)
    {
        pch = my_strstr(&d_file[i*LINE_WIDTH], Regexp);
        if(pch != NULL)
        {
            my_memcpy(&result[offset*LINE_WIDTH], &d_file[i*LINE_WIDTH], sizeof(char)*LINE_WIDTH);
            offset++;
        }
    }

    for(i = 0; i < FILE_LINE; i++)
    {
        if(&result[i*LINE_WIDTH]!=NULL)
            printf("%s", &result[i*LINE_WIDTH]);
    }

    return 0;
}

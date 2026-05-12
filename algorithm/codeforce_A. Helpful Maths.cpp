#include <stdio.h>
#include <string.h>
int main(void)
{
    char str[101] = {0,};
    int num[3] = {0,};
    scanf("%s", str);
    int len = strlen(str);
    int cnt = 0;
    for (int i = 0; i < len; i++)
    {
        if(str[i] == '1')
            num[0]++;
        else if(str[i] == '2')
            num[1]++;
        else if(str[i] == '3')
            num[2]++;
        else
            continue;
    }
    cnt = num[0] + num[1] + num[2];

    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < num[i]; j++)
        {
            printf("%d", i+1);
            if(cnt > 1)
                printf("+");
            cnt--;
        }
    }
    printf("\n");
}
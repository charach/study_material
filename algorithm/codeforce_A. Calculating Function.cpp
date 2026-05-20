#include <stdio.h>

int main(void)
{
    long long n;
    scanf("%llu", &n);

    if (n % 2 == 0)
        printf("%lld\n", n / 2);
    else
        printf("%lld\n", ((1 + n) / 2) * -1);

}
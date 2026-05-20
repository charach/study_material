#include <stdio.h>

#define MIN(a,b) ((a)<(b)?(a):(b))

unsigned long long GCD(unsigned long long a, unsigned long long b){
    if (b == 0) return a;
    return GCD(b, a % b);
}

int main(void){
    unsigned long long a,b,x,y;

    scanf("%llu %llu %llu %llu", &a, &b, &x, &y);

    unsigned long long gcd = GCD(x,y);
    unsigned long long tmpX = x / GCD(x,y);
    unsigned long long tmpY = y / GCD(x,y);

    unsigned long long ans = MIN(a / tmpX, b / tmpY);
    printf("%llu\n", ans);
    return 0;
}
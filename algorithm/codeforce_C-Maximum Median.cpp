#include <iostream>
#include <algorithm>

using namespace std;

int main(void)
{
    int n;
    long long k;
    cin >> n >> k;

    long long a[n];
    for (int i = 0; i < n; i++)
        cin >> a[i];

    sort(a, a + n);  // 오름차순 정렬

    // median은 인덱스 n/2. median을 올리려면 상위 절반(인덱스 n/2..n-1)만 신경 쓴다.
    // median부터 위로 올리며, 같이 올려야 하는 원소 개수(cnt)를 1씩 늘려간다.
    int mid = n / 2;
    long long cur = a[mid];

    for (int i = mid + 1; i < n && k > 0; i++) {
        long long cnt = i - mid;          // 현재 cur 높이에 있는 원소 개수
        long long diff = a[i] - cur;      // 다음 단계(a[i])까지 올려야 하는 높이
        long long cost = diff * cnt;      // cnt개를 diff만큼 올리는 비용

        if (cost <= k) {
            k -= cost;
            cur = a[i];                   // 다음 단계까지 도달
        } else {
            cur += k / cnt;               // 예산 한도까지만 올림
            k = 0;
        }
    }

    // 맨 위(상위 절반이 전부 cur로 평평)까지 도달했고 예산이 남으면,
    // 남은 k를 (n/2+1)개에 골고루 나눠 median을 더 올린다.
    if (k > 0) {
        cur += k / (mid + 1);
    }

    cout << cur << endl;
    return 0;
}

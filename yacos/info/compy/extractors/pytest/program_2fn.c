int max(int a, int b) {
  if (a > b) {
    return a;
  } else {
    return b;
  }
}
int foo(int x) {
  return max(1, x);
}


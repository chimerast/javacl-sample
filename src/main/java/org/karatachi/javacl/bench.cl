__kernel void bench(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    
    float x = in[i];
    
    float total = 0.0f;
    for (int j = 0; j < 1000; ++j) {
        total += cos(x * j) * sin((float) j);
    }
    out[i] = total;
}

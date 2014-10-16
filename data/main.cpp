#include <fstream>

int main()
{
    std::ofstream os("test.bin", std::ios::binary);
    
    int v(2);
    os.write((char*)&v,sizeof(int));
}


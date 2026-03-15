#include <iostream>
#include <string>
using namespace std;

string reverseString(string str) {
    int len = str.length();
    for (int i = 0; i < len / 2; i++) {
        std::swap(str[i], str[len - i - 1]);
    }
    return str;
}

int main() {
    std::string str;
    std::cout << "Enter a string: ";
    std::cin >> str;
    std::cout << "Reversed string: " << reverseString(str) << std::endl;
    return 0;
}

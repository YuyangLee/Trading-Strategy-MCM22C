#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#define FILENAME "data.csv"

using namespace std;

double alpha[11] = { 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10 };

int main()
{
    for (int i = 0; i < 11; i++)
    {
        double btc = 0.0, cash = 1000.0, value = 1000.0;
        int day = 1;

        ifstream file(FILENAME, ios::in);
        string line;
        getline(file, line);

        while (getline(file, line))
        {
            stringstream ss(line);
            string str;
            double price_btc, diff;
            for (int j = 0; j < 7; j++)
            {
                getline(ss, str, ',');
                if (j == 1)
                {
                    price_btc = stod(str);
                }
            }
            diff = stod(str);
            //cout << price_btc << " " << diff << endl;

            if (diff > 0 && cash > 0 && diff / price_btc > alpha[i])
            {
                // buy
                btc += cash / (price_btc * 1.02);
                cash = 0.0;
            }
            else if (diff < 0 && btc > 0 && diff / price_btc < -alpha[i])
            {
                // sell
                cash += btc * price_btc * 0.98;
                btc = 0.0;
            }

            value = price_btc * btc + cash;
            //cout << day++ << "  " << price_btc << "  " << value << "  " << cash << "  " << btc << endl;
        }

        cout << "alpha:" << alpha[i] << "  " << "value:" << value << endl;
    }

    return 0;
}

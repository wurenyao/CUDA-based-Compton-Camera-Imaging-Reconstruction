#include "../include/Header.cuh"

// used functions
//****************************************************************************************************//
double GetLength(double z1, double y1, double x1, double z2, double y2, double x2)
{
    double len = sqrt(pow((x1-x2), 2.0)+pow((y1-y2), 2.0)+pow((z1-z2), 2.0));
    return len;
}
//****************************************************************************************************//
void Print_Time_Complete( double clock_start , double clock_end , bool fin = 0 ){
  double total_time =  (clock_end - clock_start)/(CLOCKS_PER_SEC);
  unsigned minutes = total_time/60;
  double seconds = total_time - minutes*60;

  string pref = " -- Time Taken =";
  string post = "            \n -- Done\n";

  if ( fin == 1 ){
    pref = "\nReconstruction Complete \nTotal Runtime =";
    post = "            \n\n";
  }

  if ( minutes > 0 ) printf( "%s %4u minutes and %4.2f seconds %s" , pref.c_str() , minutes , seconds , post.c_str() );
  else               printf( "%s %4.2f seconds %s" , pref.c_str() , seconds , post.c_str() );
}
//****************************************************************************************************//

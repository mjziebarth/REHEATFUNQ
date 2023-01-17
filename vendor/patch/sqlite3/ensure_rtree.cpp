/*
 * This file tests whether the `sqlite3_rtree_query_callback` function
 * is available after the sqlite3 install.
 */

#include <sqlite3.h>

static int xQueryFunc(sqlite3_rtree_query_info*){
	return 0;
}

int main(){
	volatile int res =
	   sqlite3_rtree_query_callback(nullptr, nullptr, xQueryFunc, nullptr,
	                                nullptr);
}

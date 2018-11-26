#define BKTREE_STRINGS_SIZE 4096
#define BKTREE_TREE_SIZE 1147483648

#define BKTREE_STRING_MAX 24

#define BKTREE_OK 0
#define BKTREE_FAIL 1

#define BKTREE_GET_STRING(bktree, string_offset) (bktree->strings + string_offset + 1)

#define BKTREE_GET_STRING_LEN(bktree, string_offset) (*(bktree->strings + string_offset))

#define BKTREE_GET_STRING_WCS(bktree, string_offset) (bktree->strings_wcs + string_offset + 1)

#define BKTREE_GET_STRING_LEN_WCS(bktree, string_offset) (*(bktree->strings_wcs + string_offset))

#include <string>
#include <vector>

typedef struct {
    long string_offset;
    int next[BKTREE_STRING_MAX];
} BKNode;

typedef struct {
    int size;

    BKNode * tree;
    BKNode * tree_cursor;
    size_t tree_size;

	char * strings;
	char * strings_cursor;
	wchar_t * strings_wcs;
	wchar_t * strings_cursor_wcs;
	size_t strings_size;

    // word1, len(word1), word2, len(word2), max
	int(*distance)(char *, int, char *, int, int);
	int(*distance_wcs)(wchar_t *, int, wchar_t *, int, int);
} BKTree;

struct BKResult_s {
    int distance;
	std::string str;
	std::wstring str_wcs;
	//struct BKResult_s * next;
};
typedef struct BKResult_s BKResult;


BKTree * bktree_new(int(*distance)(char *, int, char *, int, int));
BKTree * bktree_new_wcs(int(*distance)(wchar_t *, int, wchar_t *, int, int));
void bktree_destroy(BKTree * bktree);
BKNode * bktree_add(BKTree * bktree, char * string, unsigned char len);
BKNode * bktree_add_wcs(BKTree * bktree, wchar_t * string, unsigned char len);
void bktree_node_print(BKTree * bktree, BKNode * node);

std::vector<BKResult> bktree_query(BKTree * bktree, char * string, unsigned char len, int max);
std::vector<BKResult> bktree_query_wcs(BKTree * bktree, wchar_t * string, unsigned char len, int max);
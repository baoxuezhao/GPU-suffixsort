#include <string>
#include <iostream>
#include <sstream>

using std::string;
using std::cout;
using std::endl;

#include "../inc/Ref.h"
#include "../inc/Hashtable.h"
#include "../inc/Config.h"
#include "../inc/Suffix.h"
#include "../inc/sufsort_util.h"

void testRef(char* argv[])
{
	string file_name(argv[1]);
	Ref *human_ref = Ref::Instance(file_name);
	human_ref->CompactRef();
	human_ref->StoreReference();
	human_ref->GetRepeatInfo();
	human_ref->OutputRefInfo();
	delete human_ref;

/*
	string config_file_path(argv[1]);
	string org_ref_path(argv[2]);

	Config config(config_file_path);
	config.OutputConfigInfo();
	Ref* human_ref1 = Ref::Instance(config);
	Ref* human_ref2 = Ref::Instance(org_ref_path);
	
	human_ref2->CompactRef();
	if (human_ref1->PackedRefEqual(human_ref2))
		cout << "packed references of human_ref1 and human_ref2 are the same" << endl;
	else
		cout << "error: packed references of human_ref1 and human_ref2 are different" << endl;
	delete human_ref1;
	delete human_ref2;	
*/	
}

void testHash(char* argv[])
{
	string file_name(argv[1]);
	uint32 init_k = get_num(argv[2]);
	uint64 ref_size;
	Ref* human_ref = Ref::Instance(file_name);
	human_ref->CompactRef();
	ref_size = human_ref->GetRefSize();
	Hashtable* hashtable1 = Hashtable::Instance(human_ref);
	Hashtable* hashtable2 = Hashtable::Instance(human_ref);
	
	cout << "inserting values into hash table1..." << endl;
	hashtable1->InsertAllSuffixes(human_ref, init_k);
	cout << endl;
	cout << "finished "<< endl;
	
	cout << "inserting values into hash table2..." << endl;
	hashtable2->InsertAllSuffixesParallel(human_ref, init_k);
	cout << endl;
	cout << "finished "<< endl;
	
	if (hashtable1->ApproEqual(hashtable2))
		cout << "paralleled insertion is correct" << endl;
	else
		cout << "paralleled insertion is incorrect" << endl;

/*	
	cout << "collecting hash table information..." << endl;
	hashtable->OutputHashtableInfoExcludeConflict();
	cout << "finish collecting hash table data" << endl;
*/	
	cout << "free resources..."<< endl;
	delete hashtable1;
	delete hashtable2;
	delete human_ref;
	cout << "finished "<< endl;
}

void compareWithSTL(bool read_from_disk)
{
	Hashtable* hashtable = Hashtable::Instance();
	hashtable->ComparePerformanceWithSTLHash(read_from_disk);
	delete hashtable;
}

void testSuffix(int argc, char* argv[])
{
	if (argc < 3)
	{
		fprintf(stderr, "error: comand line argument is not correct!\n");
		fprintf(stderr, "usage:\n./gpu_sufsort file_path|config file path  ratio\n\n");
		exit(-1);
	}
	
	string file_name(argv[1]);
	Ref *sample_ref = Ref::Instance(file_name);
	stringstream ss(stringstream::in | stringstream::out);
	float ratio;
	ss << argv[2];
	ss >> ratio;

/*
	string config_file_path(argv[1]);
	Config config(config_file_path);
	config.OutputConfigInfo();
	Ref* sample_ref = Ref::Instance(config);
	sample_ref->TestPackResult();
*/	
	Suffix *suf = Suffix::Instance(sample_ref, 8);
	suf->Sort(ratio);
	
	delete sample_ref;
	delete suf;

}

int main(int argc, char* argv[])
{
//	testRef(argv);
//	testHash(argv);
//	compareWithSTL(get_bool(argv[1]));
	
	/*load packed reference*/
	testSuffix(argc, argv);

	/*	
	string file_name(argv[1]);
	string file_name2(argv[2]);
	Ref *input1 = Ref::Instance(file_name);
	Ref *input2 = Ref::Instance(file_name2);
	uint8 *buf1 = input1->GetRefBuffer();
	uint8 *buf2 = input2->GetRefBuffer();
	for (int i = 0; i < input1->GetRefSize(); i++)
		if (buf1[i] != buf2[i])
			printf("index: %d, %d %d\n", i, buf1[i], buf2[i]);
	delete input1;
	delete input2;
	*/
	return 0;
}

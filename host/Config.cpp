#include <fstream>
#include <iostream>
#include <string>
using namespace std;

#include "../inc/Config.h"
#include "../inc/sufsort_util.h"

Config::Config(const string& file_path):  data_path(""), reference_name(""), reference_packed_name("")
{
	config_file_name = get_filename_from_path(file_path);
	alphabet_size = 0;
	ref_size = 0;
	ref_packed_size = 0;
	bits_per_ch = 0;
	ch_per_uint = 0;
	can_pack = false;
	num_packed = 0;
	ReadConfigFile(file_path);
}

/**
 * Users are not recommanded to modify the config file
 */
void Config::ReadConfigFile(const string& file_path)
{
	ifstream infile;
	string line_data;
	size_t found;
	infile.open(file_path.c_str(), ios::in);
	if (!infile)
		output_err_msg(FILE_OPEN_ERROR);
	while (getline(infile, line_data))
	{
		if (string::npos != line_data.find("alphabet_size"))
		{
			found = line_data.find_first_of(":");
			if (found == string::npos)
				output_err_msg(CONFIG_FORMAT_INCORRECT);
			alphabet_size = get_num(line_data.substr(found+1));
		}
		else if (string::npos != line_data.find("ref_size"))
		{
			found = line_data.find_first_of(":");
			if (found == string::npos)
				output_err_msg(CONFIG_FORMAT_INCORRECT);
			ref_size = get_num(line_data.substr(found+1));
		}
		else if (string::npos != line_data.find("ref_packed_size"))
		{
			found = line_data.find_first_of(":");
			if (found == string::npos)
				output_err_msg(CONFIG_FORMAT_INCORRECT);
			ref_packed_size = get_num(line_data.substr(found+1));
		}
		else if (string::npos != line_data.find("bits_per_ch"))
		{
			found = line_data.find_first_of(":");
			if (found == string::npos)
				output_err_msg(CONFIG_FORMAT_INCORRECT);
			bits_per_ch  = get_num(line_data.substr(found+1));
		}
		else if (string::npos != line_data.find("ch_per_uint"))
		{
			found = line_data.find_first_of(":");
			if (found == string::npos)
				output_err_msg(CONFIG_FORMAT_INCORRECT);
			ch_per_uint  = get_num(line_data.substr(found+1));
		}
		else if (string::npos != line_data.find("can_pack"))
		{
			found = line_data.find_first_of(":");
			if (found == string::npos)
				output_err_msg(CONFIG_FORMAT_INCORRECT);
			can_pack  = get_bool(line_data.substr(found+1));
		}
		else if (string::npos != line_data.find("data_path"))
		{
			found = line_data.find_first_of(":");
			if (found == string::npos)
				output_err_msg(CONFIG_FORMAT_INCORRECT);
			
			data_path  = trim(line_data.substr(found+1));
		}
		else if (string::npos != line_data.find("reference_name"))
		{
			found = line_data.find_first_of(":");
			if (found == string::npos)
				output_err_msg(CONFIG_FORMAT_INCORRECT);
			reference_name  = trim(line_data.substr(found+1));
		}
		else if (string::npos != line_data.find("reference_packed_name"))
		{
			found = line_data.find_first_of(":");
			if (found == string::npos)
				output_err_msg(CONFIG_FORMAT_INCORRECT);
			reference_packed_name  = trim(line_data.substr(found+1));
		}
		else if (string::npos != line_data.find("reference_rearranged_name"))
		{
			found = line_data.find_first_of(":");
			if (found == string::npos)
				output_err_msg(CONFIG_FORMAT_INCORRECT);
			reference_rearranged_name  = trim(line_data.substr(found+1));
		}
		else if (string::npos != line_data.find("config_file_name"))
			continue;
		else if (string::npos != line_data.find("num_packed"))
		{
			found = line_data.find_first_of(":");
			if (found == string::npos)
				output_err_msg(CONFIG_FORMAT_INCORRECT);
			num_packed  = get_num(line_data.substr(found+1));
		}
		else
		{
			cout << line_data << " ";
			output_err_msg(CONFIG_UNKNOWN_OPTION);
		}
	}

	infile.close();
}

void Config::OutputConfigInfo()
{
	cout << "------------Configuration file information---------------" << endl;
	cout << "configuration file name: " << config_file_name << endl;
	cout << "data directory: " << data_path << endl;
	cout << "reference file name: " << reference_name << endl;
	cout << "reference can be packed: " << (can_pack?"yes":"no") << endl;
	if (can_pack)
		cout << "packed reference file name: " << reference_packed_name << endl;
	else
		cout << "packed reference file name: not available" << endl;

	cout << "alphabet size: " << alphabet_size << endl;
	cout << "reference size: " << ref_size << endl;
	if (can_pack)
	{
		cout << "packed reference size: " << ref_packed_size  << endl;
		cout << "number of bits per character of packed reference: " << bits_per_ch << endl;
		cout << "number of characters in each packed unit: " << ch_per_uint << endl;
		cout << "number of packed references: " << num_packed << endl;
	}
	else
	{
		cout << "packed reference size: not available"  << endl;
		cout << "number of bits per character of packed reference: not available" << endl;
		cout << "number of characters in each packed unit: not available" << endl;
	}
}

#include <string>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <map>
#include <iomanip>
#include <time.h>
using namespace std;



//======================= I/O related ==========================//
//-------- utility ------//
void getBaseName(string &in,string &out,char slash,char dot)
{
	int i,j;
	int len=(int)in.length();
	for(i=len-1;i>=0;i--)
	{
		if(in[i]==slash)break;
	}
	i++;
	for(j=len-1;j>=0;j--)
	{
		if(in[j]==dot)break;
	}
	if(j==-1)j=len;
	out=in.substr(i,j-i);
}
void getRootName(string &in,string &out,char slash)
{
	int i;
	int len=(int)in.length();
	for(i=len-1;i>=0;i--)
	{
		if(in[i]==slash)break;
	}
	if(i<=0)out=".";
	else out=in.substr(0,i);
}

//=================== upper and lower case ====================//
//----------upper_case-----------//
void toUpperCase(char *buffer)
{
	for(int i=0;i<(int)strlen(buffer);i++)
	if(buffer[i]>=97 && buffer[i]<=122) buffer[i]-=32;
}
void toUpperCase(string &buffer)
{
	for(int i=0;i<(int)buffer.length();i++)
	if(buffer[i]>=97 && buffer[i]<=122) buffer[i]-=32;
}
//----------lower_case-----------//
void toLowerCase(char *buffer)
{
	for(int i=0;i<(int)strlen(buffer);i++)
	if(buffer[i]>=65 && buffer[i]<=90) buffer[i]+=32;
}
void toLowerCase(string &buffer)
{
	for(int i=0;i<(int)buffer.length();i++)
	if(buffer[i]>=65 && buffer[i]<=90) buffer[i]+=32;
}


//--------- FASTA I/O ------------//
//FASTA
int ReadToFile_FASTA(string &fn,vector<pair<int, int> > &alignment,
					  string &nam1_content,string &nam2_content,
					  string &nam1_full,string &nam2_full,
					  string &nam1,string &nam2)
{
	int i;
	int cur1=0;
	int cur2=0;
	int len;
	int len1,len2;
	alignment.clear();
	//init
	string seq="";  //sequence
	string tmp="";  //template
	//load
	ifstream fin;
	string buf,temp;
	fin.open(fn.c_str(), ios::in);
	if(fin.fail()!=0)
	{
		fprintf(stderr,"alignment file not found [%s] !!!\n",fn.c_str());
		return -1;
	}
	//read tmp
	for(;;)
	{
		if(!getline(fin,buf,'\n'))goto badend;
		len=(int)buf.length();
		if(len>1)
		{
			if(buf[0]=='>')
			{
				istringstream www(buf);
				www>>temp;
				len=(int)temp.length();
				nam1=temp.substr(1,len-1);
				break;
			}
		}
	}
	for(;;)
	{
		if(!getline(fin,buf,'\n'))goto badend;
		len=(int)buf.length();
		if(len==0)continue;
		if(len>1)
		{
			if(buf[0]=='>')
			{
				istringstream www(buf);
				www>>temp;
				len=(int)temp.length();
				nam2=temp.substr(1,len-1);
				break;
			}
		}
		tmp+=buf;
	}
	//read seq
	for(;;)
	{
		if(!getline(fin,buf,'\n'))break;
		len=(int)buf.length();
		if(len==0)continue;
		seq+=buf;
	}
	//process
	len1=(int)seq.length();
	len2=(int)tmp.length();
	if(len1!=len2)
	{
		fprintf(stderr,"alignment len not equal [%s] !!!\n",fn.c_str());
		return -1;
	}
	len=len1;
	nam1_content.clear();
	nam2_content.clear();
	for(i=0;i<len;i++)
	{
		if(tmp[i]!='-' && seq[i]!='-') //match
		{
			nam1_content.push_back(tmp[i]);
			nam2_content.push_back(seq[i]);
			cur1++;
			cur2++;
			alignment.push_back(pair<int,int>(cur1,cur2));
		}
		else
		{
			if(tmp[i]!='-') //Ix
			{
				nam1_content.push_back(tmp[i]);
				cur1++;
				alignment.push_back(pair<int,int>(cur1,-cur2));
			}
			if(seq[i]!='-') //Iy
			{
				nam2_content.push_back(seq[i]);
				cur2++;
				alignment.push_back(pair<int,int>(-cur1,cur2));
			}
		}
	}
	//return
	nam1_full=tmp;
	nam2_full=seq;
	return 1; //success

badend:
	fprintf(stderr,"alignment file format bad [%s] !!!\n",fn.c_str());
	return -1;
}

//============= get file related ==========//
int WS_Get_File(string &filename,vector <string> &output,int skip=0)
{
	ifstream fin;
	string wbuf,temp;
	fin.open(filename.c_str(), ios::in);
	if(fin.fail()!=0)
	{
		fprintf(stderr,"%s not found!\n",filename.c_str());
		exit(-1);
	}
	//skip
	for(int i=0;i<skip;i++)
	{
		if(!getline(fin,wbuf,'\n'))
		{
			fprintf(stderr,"FORMAT BAD AT FEATURE FILE %s \n",filename.c_str());
			exit(-1);
		}
	}
	//get
	output.clear();
	int count=0;
	for(;;)
	{
		if(!getline(fin,wbuf,'\n'))break;
		output.push_back(wbuf);
		count++;
	}
	return count;
}

//============ ws comparison two fasta file ===========//
void WS_Comp_Two_FASTA(string &fasta1,string &fasta2,int partial_num=4,int check=1)
{
	//load file A
	string nam1_content_A,nam2_content_A,nam1_full_A,nam2_full_A,nam1_A,nam2_A;
	vector<pair<int, int> > alignment_A;
	int retv1=ReadToFile_FASTA(fasta1,alignment_A,nam1_content_A,nam2_content_A,nam1_full_A,nam2_full_A,nam1_A,nam2_A);
	if(retv1!=1)
	{
		fprintf(stderr,"ali_fasta_A not found!! [%s]\n",fasta1.c_str());
		exit(-1);
	}
	//load file B
	string nam1_content_B,nam2_content_B,nam1_full_B,nam2_full_B,nam1_B,nam2_B;
	vector<pair<int, int> > alignment_B;
	int retv2=ReadToFile_FASTA(fasta2,alignment_B,nam1_content_B,nam2_content_B,nam1_full_B,nam2_full_B,nam1_B,nam2_B);
	if(retv2!=1)
	{
		fprintf(stderr,"ali_fasta_B not found!! [%s]\n",fasta2.c_str());
		exit(-1);
	}
	//check content
	if(check!=1)
	{
		if(nam1_content_A!=nam1_content_B)
		{
			fprintf(stderr,"nam1_content_A != nam1_content_B \n");
			fprintf(stderr,"%s\n",nam1_content_A.c_str());
			fprintf(stderr,"%s\n",nam1_content_B.c_str());
			exit(-1);
		}
		if(nam2_content_A!=nam2_content_B)
		{
			fprintf(stderr,"nam2_content_A != nam2_content_B \n");
			fprintf(stderr,"%s\n",nam2_content_A.c_str());
			fprintf(stderr,"%s\n",nam2_content_B.c_str());
			exit(-1);
		}
	}
	else
	{
		if(nam1_content_A.size()!=nam1_content_B.size())
		{
			fprintf(stderr,"size nam1_content_A != nam1_content_B, %d!=%d \n",
				(int)nam1_content_A.size(),(int)nam1_content_B.size());
			exit(-1);
		}
		if(nam2_content_A.size()!=nam2_content_B.size())
		{
			fprintf(stderr,"size nam2_content_A != nam2_content_B, %d!=%d \n",
				(int)nam2_content_A.size(),(int)nam2_content_B.size());
			exit(-1);
		}
	}
	//compare two fasta files
	int i;
	int ii,jj;
	int lali_A=0;
	int lali_B=0;
	int size_A=(int)alignment_A.size();
	int size_B=(int)alignment_B.size();
	int moln2_A=(int)nam2_content_A.size();
	int moln2_B=(int)nam2_content_B.size();
	int *ali2_A=new int[moln2_A];
	int *ali2_B=new int[moln2_B];
	for(i=0;i<moln2_A;i++)ali2_A[i]=-1;
	for(i=0;i<moln2_B;i++)ali2_B[i]=-1;
	for(i=0;i<size_A;i++)
	{
		ii=alignment_A[i].first;
		jj=alignment_A[i].second;
		if(ii>0&&jj>0)
		{
			ali2_A[jj-1]=ii-1;
			lali_A++;
		}
	}
	for(i=0;i<size_B;i++)
	{
		ii=alignment_B[i].first;
		jj=alignment_B[i].second;
		if(ii>0&&jj>0)
		{
			ali2_B[jj-1]=ii-1;
			lali_B++;
		}
	}
	//final comparison (use the second as the standard)
	int identical=0;
	int partial_order=0;
	for(i=0;i<moln2_B;i++)
	{
		if(ali2_A[i]!=-1 && ali2_B[i]!=-1)
		{
			if(ali2_A[i]==ali2_B[i])identical++;
			if(abs(ali2_A[i]-ali2_B[i])<=partial_num)partial_order++;
		}
	}
	//output
	if(lali_A==0 || lali_B==0)
	{
		printf("%s %s 0 0 %d %d 0 0 0 0 \n",
			nam1_B.c_str(),nam2_B.c_str(),lali_A,lali_B);
	}
	else
	{
		printf("%s %s %d %d %d %d %lf %lf %lf %lf \n",
			nam1_B.c_str(),nam2_B.c_str(),identical,partial_order,lali_A,lali_B,
			1.0*identical/lali_A,1.0*identical/lali_B,1.0*partial_order/lali_A,1.0*partial_order/lali_B);
	}
	//delete
	delete [] ali2_A;
	delete [] ali2_B;
}



//----------- main -------------//
int main(int argc,char **argv)
{

	//---- Alignment_Comparison ----//
	{
		if(argc<4)
		{
			printf("Usage: Alignment_Comparison <fasta1> <fasta2> [partial_num=4] \n");
      printf("\n");
      printf("The screen output will be 10 colums\n");
      printf("1st column is the template name.\n");
      printf("2nd column is the query name.\n");
      printf("3rd column is the exact identical match (denoted as identical).\n");
      printf("4th column is the partial_order match.\n");
      printf("5th column is the length of alignment, denoted as Lali_A \n");
      printf("6th column is the Length of ground-truth alignment alignemnt, denoted as Lali_B. \n");
      printf("7th column is precision, defined as 1.0*identical/lali_A. \n");
      printf("8th column is recall, defined as 1.0*identical/lali_B. \n");
      printf("9th column is defined as 1.0*partial_order/lali_A. \n");
      printf("10th column is defined as 1.0*partial_order/lali_B. \n");
			exit(-1);
		}
		string fasta1=argv[1];
		string fasta2=argv[2];
		int partial_num=atoi(argv[3]);
		WS_Comp_Two_FASTA(fasta1,fasta2,partial_num);
		exit(0);
	}

}

/*
	Program:	nMHDust
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		10/13/10
	Purpose:	This is C version of the nMHDust code based on the paper by
			Schroer and Kopp "A three-fluid system of equations describing
			dusty magnetoplasmas with dynamically important dust and ion
			components."   Physics of Plasmas
			It includes a neutral component with ionization and recombination
			rates.  Data is output in the netCDF common data format.
			Compile with:
				gcc nmhdust.c -lm -lnetcdf -o nmhdust
                        On cluster:
				gcc nmhdust.c -lnetcdf -O2 -L$NETCDFHOME/lib -I$NETCDFHOME/include -lm -o nmhdust
				pathcc -lnetcdf -lm -O2 -L$NETCDFHOME/lib -I$NETCDFHOME/include nmhdust.c -o nmhdust
				gcc nmhdust.c -lnetcdf -fopenmp -O2 -L$NETCDFHOME/lib -I$NETCDFHOME/include -lm -o nmhdust_omp
				pathcc -lnetcdf -lm -openmp -O2 -L$NETCDFHOME/lib -I$NETCDFHOME/include nmhdust.c -o nmhdust_omp

*/
/*Header Files*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <netcdf.h>
#include <string.h>
#include <time.h>
/*OpenMP Header*/
#ifdef _OPENMP
	#include <omp.h>
	#define TIMESCALE 1
#else
	#define omp_get_thread_num() 0
	#define omp_get_num_procs() 0
	#define omp_get_num_threads() 1
	#define omp_set_num_threads(bob) 0
	#define omp_get_wtime() clock()
	#define TIMESCALE CLOCKS_PER_SEC
#endif
/*Preprocessor Values - Constants*/
#define		NA 14
#define		NP 8
/*Structures Block*/
typedef struct
{
	double	***rho;
	double	***rhoi;
	double	***rhoe;
	double	***rhon;
	double	***sx;
	double	***sy;
	double	***sz;
	double	***six;
	double	***siy;
	double	***siz;
	double	***sex;
	double	***sey;
	double	***sez;
	double	***snx;
	double	***sny;
	double	***snz;
	double	***p;
	double	***pi;
	double	***pe;
	double	***pn;
	double	***bx;
	double	***by;
	double	***bz;
	double	***jx;
	double	***jy;
	double	***jz;
	double  ***ex;
	double  ***ey;
	double  ***ez;
	double	***zd;
	double	***zi;
	double	***gamma;
	double	***gammai;
	double	***gammae;
	double	***gamman;
	double	***nuid;
	double  ***nuie;
	double	***nude;
	double	***nudn;
	double	***nuin;
	double	***nuen;
	double  ***gravx;
	double  ***gravy;
	double  ***gravz;
	unsigned short int     ***rhosmo;
	unsigned short int     ***rhoismo;
	unsigned short int     ***rhonsmo;
	unsigned short int	***sxsmo;
	unsigned short int     ***sysmo;
	unsigned short int     ***szsmo;
	unsigned short int     ***sixsmo;
	unsigned short int     ***siysmo;
	unsigned short int     ***sizsmo;
	unsigned short int     ***snxsmo;
	unsigned short int     ***snysmo;
	unsigned short int     ***snzsmo;
	unsigned short int     ***psmo;
	unsigned short int     ***pismo;
	unsigned short int     ***pesmo;
	unsigned short int     ***pnsmo;
	double	visx;
	double   visy;
	double   visz;
	double	md;
	double	mi;
	double	me;
	double	mn;
	double	ioniz;
	double	recom;
	double   e;
} MHD;
typedef struct
{
	double	***x;
	double	***y;
	double	***z;
	double	***difx;
	double	***dify;
	double  ***difz;
	double	***ddifx;
	double	***ddify;
	double	***ddifz;
	double	***ddifmx;
	double	***ddifmy;
	double	***ddifmz;
	double	***ddifpx;
	double	***ddifpy;
	double	***ddifpz;
	double	***meanmx;
	double	***meanmy;
	double	***meanmz;
	double	***meanpx;
	double	***meanpy;
	double	***meanpz;
	double  ***ssx;
	double  ***ssy;
	double  ***ssz;
	double  ***s;
	double  ***bx0;
	double  ***by0;
	double  ***bz0;
	double	dt;
	double	xmin;
	double	xmax;
	double	ymin;
	double	ymax;
	double	zmin;
	double	zmax;
	int	nx;
	int	ny;
	int	nz;
	int	nxm1,nym1,nzm1,nxp1,nyp1,nzp1; //Array Index Helpers nxm1=nx-1 nxp1=nx+1
	int	nxm2,nym2,nzm2,nxp2,nyp2,nzp2; //Array Index Helpers nxm1=nx-2 nxp1=nx+2
	int	nxm3,nym3,nzm3,nxp3,nyp3,nzp3; //Array Index Helpers nxm1=nx-3 nxp1=nx+3
	int	nxm4,nym4,nzm4,nxp4,nyp4,nzp4; //Array Index Helpers nxm1=nx-1 nxp1=nx+4
	unsigned short int	econt;
	unsigned short int 	perx;
	unsigned short int	pery;
	unsigned short int	perz;
        unsigned short int     divbsolve;
	unsigned short int     ballon;
	unsigned short int     perturb;
	int	movieout;
	int     lastball;
	unsigned short int	chksmooth;
	unsigned short int     kdo,kio;
	unsigned short int	dnudi,dnude,dnuie,dnudn,dnuin,dnuen;
        unsigned short int     verb;
        unsigned short int	codetype;  // 0: DENISIS; 1: nMHDust_leap;
	int	nout;
	int	maxntime;
// Index 1: 0=xmax,1=xmin,2=ymax,3=ymin,4=zmax,5=zmin
// Index 2: 0=bx,1=by,2=bz,3=sx,4=sy,5=sz,6=six,7=siy,8=siz (for k)
// Index 2: 0=rho,1=rhoi,2=rhoe,3=rhoe,4=p,5=pi,6=pe,7=pn,8=bx,9=by,10=bz,11=sx,12=sy,13=sz,14=six,15=siy,16=siz,17=snx,18=sny,19=snz  (for a and d)
	int	k[6][NA];
	int	a[6][NA];
	double	d[6][NA];
	double	c1[6][NA];
	double	c2[6][NA];
	double	c3a[6][NA];
	double	c3b[6][NA];
} GRID;
/* Protype Block */
void	initial_conditions(MHD *,GRID *,int *,int *,char *);
void	makegrid(GRID *,MHD *,int,int,int,int,int,int,double,double,double);
void	bcorg_nmhdust(MHD *,GRID *,double *);
void	first_step(MHD *, GRID *,MHD *);
void	intscheme_nmhdust(MHD *,GRID *,int *,double *);
void	leap_nmhdust(MHD *,GRID *,short int);
void    smooth(double ***,unsigned short int ***,GRID *,int);
void    valcheck(MHD *,GRID *,int*);
void    divbsolver(MHD *,GRID *);
void	nudicalc(MHD *,GRID *);
void    nudecalc(MHD *,GRID *);
void    nuiecalc(MHD *,GRID *);
void    nudncalc(MHD *,GRID *);
void    nuincalc(MHD *,GRID *);
void    nuencalc(MHD *,GRID *);
void	output(MHD *,GRID *,int,double,char *);
void	netcdf_input(MHD *, GRID *,char *);
void    ball(MHD *,GRID *,int *,double *);
void    energy(MHD *,GRID *,double);
void	perturb(MHD *, GRID *, double *);
void	movie(MHD *,GRID *,double);
void    handle_error(int);
void    allocate_grid(MHD *, GRID *);
void*** newarray(int, int, int, int);
double  min3d(double ***,int,int,int);
double  max3d(double ***,int,int,int);

/* Main Function*/
main(int argc, char **argv)
{

	int 		ntime,maxntime,nout,i;
	double 		time;
	char		*configfile,*runname,*netcdfext,*filename;
	GRID		*grid;
	MHD		*mhd;

	// Initialize some values
	time=0.0;
	ntime=0;
	maxntime=0;
	netcdfext=".nc";

	// Allocate MHD, Grid, and helper
	mhd    = (MHD *)  malloc(sizeof(MHD));
	grid   = (GRID *) malloc(sizeof(GRID));

	//  Print some OpenMP stuff if it's running
	omp_set_num_threads(16);
	#pragma omp parallel
	{
		#pragma omp single
		{
			if (omp_get_num_threads() > 1)
			{
				printf("===== n M H D u s t (Open MP) =====\n");
				printf("    -Processors: %d\n",omp_get_num_procs());
				printf("    -Threads:    %d\n",omp_get_num_threads());
			}
			else
			{
				printf("===== n M H D u s t =====\n");
			}
		} /*--  End Single region --*/
	} /*-- End of parallel region --*/
	/*******************************************************************
	 Basically we want to deal with four possibilities.
	  1)  User passes nothing then look for config.dat
	  2)  User passes filename.nc then do a restart from file
	  3)  User passes run_name then look for a config file by that name
	  4)  User passes -help or -h and we print a help file
	********************************************************************/
	// Default to using config.dat;
	configfile="config.dat";
	runname="data";
        // Default verb to true
        grid->verb=1;
	if (argc > 1)
	{
		for(i=1;i<argc;i++)
		{
			if (strcmp(argv[i],"-noverb")==0)
			{
				grid->verb=0;
			}
		}
	}
        if (grid->verb)
	{
		printf(" -Number of command line arguments:%d\n",argc);
		for (i=0;i<argc;i++)
		{
				printf("    argv[%1d/%1d]=%s\n",i+1,argc,argv[i]);
		}
	}
	if (argc > 1)
	{
		for(i=1;i<argc;i++)
		{
			if ((strcmp(argv[i],"-h")==0) || (strcmp(argv[i],"-help")==0))
			{
				printf("Help Message\n");
				exit(-1);
			}
			else if (strcmp(argv[i],"-nc")==0)
			{
				if (i+1<argc)
				{
					i++;
					filename=argv[i];
					//Need to extract the run name
					runname="data";
					if (grid->verb) printf(" -Loading Initial Conditions from: %s\n",filename);
					//Call the function to open the netCDF file
					printf("!!!!! ERROR: Not implemented !!!!!\n");
					exit(-1);				
				}
				else
				{
					printf("!!!!! ERROR: No netCDF File Supplied !!!!!\n");
					printf("      ./exe -nc filename.nc\n");
					exit(-1);
				}
			}
			else if (strcmp(argv[i],"-cfg")==0)
			{
				if (i+1<argc)
				{
					i++;
					if (strcmp(argv[i],"config")==0)
					{
						if (grid->verb) printf(" -config.dat detected!\n");
						configfile="config.dat";
					}
					else
					{
						configfile=argv[i];
						strcat(configfile,".dat");
						runname=argv[i];	
					}
					if (grid->verb) printf(" -Loading Initial Conditions from: %s\n",configfile);
					initial_conditions(mhd,grid,&nout,&maxntime,configfile);
				}
				else
				{
					printf("!!!!! ERROR: No Run File Supplied !!!!!\n");
					printf("      ./exe -cfg runname\n");
					exit(-1);
				}
			}
		}
	}
	else
	{
		if (grid->verb) printf(" -Loading Initial Conditions from %s\n",configfile);
		initial_conditions(mhd,grid,&nout,&maxntime,configfile);
	}
	// Now proceed having initialized the data
	grid->nout=nout;
	grid->maxntime=maxntime;
        if (grid->verb) 
	{
		if (grid->movieout > 0) printf(" --Movie Data Output to moviedata.dat at n=%06d intevals\n",grid->movieout);
		if (grid->divbsolve) printf(" --Divergence B Solver is employed\n");
		if (grid->ballon) printf(" --Ballistic Relaxation is employed\n");
		if (grid->dnudi) printf(" --Dynamic Ion-Dust Collision frequencies\n");
		if (grid->dnude) printf(" --Dynamic Dust-Electron Collision frequencies\n");
		if (grid->dnuie) printf(" --Dynamic Ion-Electron Collision frequencies\n");
		if (grid->dnudn) printf(" --Dynamic Dust-Neutral Collision frequencies\n");
		if (grid->dnuin) printf(" --Dynamic Ion-Neutral Collision frequencies\n");
		if (grid->dnuen) printf(" --Dynamic Electron-Neutral Collision frequencies\n");
	}
	// Choose Integration Path
	switch (grid->codetype)
	{
	case 0 : //DENISIS
		printf("=====  DENISIS Option Not implemented  =====\n");
		break;
	case 1 : //nMHDust
		// Apply Boundary Condition
		bcorg_nmhdust(mhd,grid,&time);
		// Do an initial output
		movie(mhd,grid,time);
		output(mhd,grid,ntime,time,runname);
		// Do first half step and second full	
		if (nout != 0)
		{
			//Integrate the even grid 1/2 timestep then the odd grid at a full timestep
			if (grid->verb) printf(" -Initial Integration\n");
			grid->dt=grid->dt/2.;
			leap_nmhdust(mhd,grid,0);
			bcorg_nmhdust(mhd,grid,&time);
			grid->dt=grid->dt*2.;
			leap_nmhdust(mhd,grid,1);
			bcorg_nmhdust(mhd,grid,&time);
			if (grid->verb) printf(" -Main Integration Loop\n");
		}
		while ((ntime < maxntime) && (nout !=0) && (ntime != -1))
		{
			intscheme_nmhdust(mhd,grid,&ntime,&time);
			if (ntime%nout == 0)
			{
				//The even grid must be brought into timestep with the odd grid.
				grid->dt=grid->dt/2.;
				leap_nmhdust(mhd,grid,0);
				bcorg_nmhdust(mhd,grid,&time);
				grid->dt=grid->dt*2.;
				//Output Data
				output(mhd,grid,ntime,time,runname);
				//Integrat the even grid 1/2 timestep then the odd grid at a full timestep
				grid->dt=grid->dt/2.;
				leap_nmhdust(mhd,grid,0);
				bcorg_nmhdust(mhd,grid,&time);
				grid->dt=grid->dt*2.;
				leap_nmhdust(mhd,grid,1);
				bcorg_nmhdust(mhd,grid,&time);
			}
			else if (ntime%222 == 0)	//Set grids to same timestep
			{
				//The even grid must be brought into timestep with the odd grid.
				grid->dt=grid->dt/2.;
				leap_nmhdust(mhd,grid,0);
				bcorg_nmhdust(mhd,grid,&time);
				grid->dt=grid->dt*2.;
				//Integrat the even grid 1/2 timestep then the odd grid at a full timestep
				grid->dt=grid->dt/2.;
				leap_nmhdust(mhd,grid,0);
				bcorg_nmhdust(mhd,grid,&time);
				grid->dt=grid->dt*2.;
				leap_nmhdust(mhd,grid,1);
				bcorg_nmhdust(mhd,grid,&time);
			}
		}
		break;
	default :
		printf("!!!!!  ERROR:  Unknown code type %02d.  Check input files.  !!!!!\n",grid->codetype);
		printf("        0 :     DENISIS\n");
		printf("        1 :     nMHDust (leapfrog)\n");
		printf("        2 :     nMHDust (leapfrog_fct)\n");
		printf("        3 :     nMHDust (mean lax wendroff)\n");
		break;
	}
	if (ntime !=-1)
	{
		if (grid->verb) printf("===== Simulation Complete =====\n");
	}
	if (ntime == -1)
	{
		output(mhd,grid,ntime,time,runname);
	}
	//Free Memory (should also free the arrays)
	//free(mhd);
	//free(grid);
}

/***********************************************************/
/* Function Blocks */
/***********************************************************/
/*
	Function:	initial_conditions
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		9/22/08
	Inputs:		struct mhd, struct grid, nout, maxn
	Outputs:	none
	Purpose:	This fuctions reads the config.dat
			file and determines how to initialize
			the data.  Includes neutrals.
			It reads config.dat for nx,ny,nz
			and calls the allocation array.
*/
void initial_conditions(MHD *mhd,GRID *grid,int *noutic,int *nendic, char *configfile)
{
	FILE	*fp,*fpr;
	char	buff[100];
	int	i,j,k,restart,tempd;
	double	tempf;
	double	gamma0,gammai0,gammae0,res0,xmin,xmax,ymin,ymax,zmin,zmax;
	double	rho0,zd0,md0,rhoi0,zi0,mi0,bx0,by0,bz0,vx0,vy0,vz0,vix0,viy0,viz0;
	double	vnx0,vny0,vnz0,nudn0,nuin0,nuen0;
	double	sx0,sy0,sz0,six0,siy0,siz0,rhoe0,p0,pi0,pe0;
	double   rhon0,pn0,snx0,sny0,snz0,gamman0,mn0,ioniz0,recom0;
	double	nuid0,nuie0,nude0,nuio0,dxmin,dymin,dzmin;
	double   radius;
	int	xeq,yeq,zeq,ldxmin,ldymin,ldzmin;
	
	// Defaults
	grid->movieout=1000;
	grid->perturb=0;
	grid->codetype=1;
//	Open and check from config.dat
	if((fp=fopen(configfile,"r")) == NULL)
	{
		printf("---ERROR:Couldn't open \"%s\" for input---\n",configfile);
		noutic=0;
		exit(-1);
	}
	printf(" --Reading \"%s\" for IC\n",configfile);
	fgets(buff,100,fp);
	sscanf(buff,"%d",&tempd);
	grid->nx=tempd;
	fgets(buff,100,fp);
	sscanf(buff,"%d",&tempd);
	grid->ny=tempd;
	fgets(buff,100,fp);
	sscanf(buff,"%d",&tempd);
	grid->nz=tempd;
	// Setup Grid Helper Indexes
	grid->nxp1=grid->nx+1;
	grid->nyp1=grid->ny+1;
	grid->nzp1=grid->nz+1;
	grid->nxm1=grid->nx-1;
	grid->nym1=grid->ny-1;
	grid->nzm1=grid->nz-1;
	grid->nxp2=grid->nx+2;
	grid->nyp2=grid->ny+2;
	grid->nzp2=grid->nz+2;
	grid->nxm2=grid->nx-2;
	grid->nym2=grid->ny-2;
	grid->nzm2=grid->nz-2;
	grid->nxp3=grid->nx+3;
	grid->nyp3=grid->ny+3;
	grid->nzp3=grid->nz+3;
	grid->nxm3=grid->nx-3;
	grid->nym3=grid->ny-3;
	grid->nzm3=grid->nz-3;
	grid->nxp4=grid->nx+4;
	grid->nyp4=grid->ny+4;
	grid->nzp4=grid->nz+4;
	grid->nxm4=grid->nx-4;
	grid->nym4=grid->ny-4;
	grid->nzm4=grid->nz-4;
	// Allocate the Arrays
	allocate_grid(mhd,grid);
	fgets(buff,100,fp);
	sscanf(buff," %d",&tempd);
	grid->codetype=tempd;
	fgets(buff,100,fp);
	sscanf(buff," %d",&tempd);
	*nendic=tempd;
	fgets(buff,100,fp);
	sscanf(buff," %d",&tempd);
	*noutic=tempd;
	fgets(buff,100,fp);
	sscanf(buff," %lf",&tempf);
	grid->dt = tempf;
	fgets(buff,100,fp);
	sscanf(buff," %d",&tempd);
	grid->divbsolve= tempd;
	fgets(buff,100,fp);
	sscanf(buff," %d",&tempd);
	grid->movieout= tempd;
	fgets(buff,100,fp);
	sscanf(buff," %d",&tempd);
	grid->econt= tempd;
	fgets(buff,100,fp);
	sscanf(buff," %d",&tempd);
	grid->ballon = tempd;
	fgets(buff,100,fp);
	sscanf(buff," %d",&tempd);
	grid->lastball = tempd;
	fgets(buff,100,fp);
	sscanf(buff," %d",&tempd);
	grid->chksmooth = tempd;
	fgets(buff,100,fp);
	sscanf(buff," %lf",&tempf);
	mhd->visx = tempf;
	fgets(buff,100,fp);
	sscanf(buff," %lf",&tempf);
	mhd->visy = tempf;
	fgets(buff,100,fp);
	sscanf(buff," %lf",&tempf);
	mhd->visz = tempf;
	fgets(buff,100,fp);
	sscanf(buff," %d",&tempd);
	grid->dnude = tempd;
	fgets(buff,100,fp);
	sscanf(buff," %d",&tempd);
	grid->dnudi = tempd;
	fgets(buff,100,fp);
	sscanf(buff," %d",&tempd);
	grid->dnuie = tempd;
	fgets(buff,100,fp);
	sscanf(buff," %d",&tempd);
	grid->dnudn = tempd;
	fgets(buff,100,fp);
	sscanf(buff," %d",&tempd);
	grid->dnuin = tempd;
	fgets(buff,100,fp);
	sscanf(buff," %d",&tempd);
	grid->dnuen = tempd;
	fgets(buff,100,fp);
	sscanf(buff," %lf",&gamma0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&gammai0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&gammae0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&gamman0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&xmin);
	grid->xmin = xmin;
	fgets(buff,100,fp);
	sscanf(buff," %lf",&xmax);
	grid->xmax = xmax;
	fgets(buff,100,fp);
	sscanf(buff," %lf",&ymin);
	grid->ymin = ymin;
	fgets(buff,100,fp);
	sscanf(buff," %lf",&ymax);
	grid->ymax = ymax;
	fgets(buff,100,fp);
	sscanf(buff," %lf",&zmin);
	grid->zmin = zmin;
	fgets(buff,100,fp);
	sscanf(buff," %lf",&zmax);
	grid->zmax = zmax;
	fgets(buff,100,fp);
	sscanf(buff," %lf",&rho0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&zd0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&md0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&rhoi0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&zi0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&mi0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&rhon0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&mn0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&bx0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&by0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&bz0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&vx0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&vy0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&vz0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&vix0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&viy0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&viz0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&vnx0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&vny0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&vnz0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&nuid0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&nuie0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&nude0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&nudn0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&nuin0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&nuen0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&ioniz0);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&recom0);
	fgets(buff,100,fp);
	sscanf(buff," %d",&xeq);
	fgets(buff,100,fp);
	sscanf(buff," %d",&yeq);
	fgets(buff,100,fp);
	sscanf(buff," %d",&zeq);
	fgets(buff,100,fp);
	sscanf(buff," %d",&ldxmin);
	fgets(buff,100,fp);
	sscanf(buff," %d",&ldymin);
	fgets(buff,100,fp);
	sscanf(buff," %d",&ldzmin);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&dxmin);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&dymin);
	fgets(buff,100,fp);
	sscanf(buff," %lf",&dzmin);
	//Skip the comentary lines ==BC==
	fgets(buff,100,fp);
	fgets(buff,100,fp);
	sscanf(buff," %d",&tempd);
	grid->perx=tempd;
	fgets(buff,100,fp);
	sscanf(buff," %d",&tempd);
	grid->pery=tempd;
	fgets(buff,100,fp);
	sscanf(buff," %d",&tempd);
	grid->perz=tempd;
	if (grid->perx > 2)
	{
		printf("--ERROR:  Check %s for BC choice.",configfile);
		printf("          perx != 0,1,2");
		noutic=0;
		return;
	}
	if (grid->pery > 2)
	{
		printf("--ERROR:  Check %s for BC choice.",configfile);
		printf("          pery != 0,1,2");
		noutic=0;
		return;
	}
	if (grid->perz > 2)
	{
		printf("--ERROR:  Check %s for BC choice.",configfile);
		printf("          perz != 0,1,2");
		noutic=0;
		return;
	}
	for(j=0;j<6;j++)
	{
		//Skip the comentary line ==L-Max==
		fgets(buff,100,fp);
		//Read the k values
		for(i=0;i<NA;i++)
		{
			fgets(buff,100,fp);
			sscanf(buff," %d",&tempd);
			grid->k[j][i]=tempd;
		}
		//Read the a values
		for(i=0;i<NA;i++)
		{
			fgets(buff,100,fp);
			sscanf(buff," %d",&tempd);
			grid->a[j][i]=tempd;
		}
		//Read the d values
		for(i=0;i<NA;i++)
		{
			fgets(buff,100,fp);
			sscanf(buff," %lf",&tempf);
			grid->d[j][i]=tempf;
		}
	}
//	Close config.dat
	fclose(fp);
//      Output values from configuration file
	if (grid->verb)
	{
		printf("      Grid Parameters\n");
		printf("      ---------------\n");
		printf("        nv=%9d   ( %7.2f MB )\n",grid->nx*grid->ny*grid->nz,(sizeof(grid)+sizeof(mhd))/1024./1024.);
		printf("        nx= %4d          ny= %4d         nz= %4d\n",grid->nx,grid->ny,grid->nz);
		printf("        x=[%3.2f,%3.2f]   y=[%3.2f,%3.2f]   z=[%3.2f,%3.2f]\n",grid->xmin,grid->xmax,grid->ymin,grid->ymax,grid->zmin,grid->zmax);
		printf("        dx=%5.4f         dy=%5.4f        dz=%5.4f\n",(grid->xmax-grid->xmin)/(grid->nx-2),(grid->ymax-grid->ymin)/(grid->ny-2),(grid->zmax-grid->zmin)/(grid->nz-2));
		printf("        dt=%7.6f\n",grid->dt);
		#ifdef _OPENMP
		printf("      Parallel Parameters\n");
		printf("      -------------------\n");
		printf("        OMP Outer loop size: %d\n",grid->nx/omp_get_num_threads());
		#endif
		printf("      Initial Values\n");
		printf("      --------------\n");
		printf("        Dust: rhod=%3.2f   zd=%+3.2f   md=%3.2f   gamma=%4.3f\n",rho0,zd0,md0,gamma0);
		printf("        Ions: rhoi=%3.2f   zi=%+3.2f   mi=%3.2f   gammai=%4.3f\n",rhoi0,zi0,mi0,gammai0);
		printf("        Elec: rhoe=%3.2f   ze=%+3.2f   me=%3.2f   gammae=%4.3f\n",mhd->mi*(zi0*rhoi0/mi0-zd0*rho0/md0)/1860.,-1.0,mhd->mi/1860.,gammae0);
		printf("        Neut: rhon=%3.2f   zn=%+3.2f   mn=%3.2f   gamman=%4.3f\n",rho0,0.0,mn0,gamman0);
		printf("        B-Field:      BX0 =%+5.4f   BY0 =%+5.4f   BZ0 =%+5.4f\n",bx0,by0,bz0);
		printf("        Velocities:   VDX0=%+5.4f   VDY0=%+5.4f   VDZ0=%+5.4f\n",vx0,vy0,vz0);
		printf("                      VIX0=%+5.4f   VIY0=%+5.4f   VIZ0=%+5.4f\n",vix0,viy0,viz0);
		printf("                      VNX0=%+5.4f   VNY0=%+5.4f   VNZ0=%+5.4f\n",vnx0,vny0,vnz0);
		printf("      Coefficients\n");
		printf("      --------------------\n");
		printf("        visx=%7.5f   visy=%7.5f   visz=%7.5f\n",mhd->visx,mhd->visy,mhd->visz);
		printf("        nuid=%7.5f   nude=%7.5f   nuie=%7.5f\n",nuid0,nude0,nuie0);
		printf("        nudn=%7.5f   nuin=%7.5f   nuen=%7.5f\n",nudn0,nuin0,nuen0);
		printf("        ioniz=%7.5f      recom=%7.5f\n",ioniz0,recom0);
	}
//	Make Grids
	if (grid->verb) printf(" --Creating Grids\n");
	makegrid(grid,mhd,xeq,yeq,zeq,ldxmin,ldymin,ldzmin,dxmin,dymin,dzmin);
//	Calc BC Coeffs
	if (grid->verb) printf(" --Calculating Boundary Value Coefficients\n");
	for(i=0;i<6;i++)
	{
		if (i==0) tempf=grid->difx[3][0][0]/grid->difx[1][0][0];
		if (i==1) tempf=grid->difx[grid->nx-4][0][0]/grid->difx[grid->nxm2][0][0];
		if (i==2) tempf=grid->dify[0][3][0]/grid->dify[0][1][0];
		if (i==3) tempf=grid->dify[0][grid->nym4][0]/grid->dify[0][grid->nym2][0];
		if (i==4) tempf=grid->difz[0][0][3]/grid->difz[0][0][1];
		if (i==5) tempf=grid->difz[0][0][grid->nzm4]/grid->difz[0][0][grid->nzm2];
		for(j=0;j<NA;j++)
		{
			grid->c1[i][j]=grid->k[i][j]+grid->a[i][j]*tempf;
			grid->c2[i][j]=-1.*grid->a[i][j]*tempf;
			if (i==0)
			{
				grid->c3a[i][j]=1.+(grid->x[0][0][0]-grid->x[1][0][0])/(grid->x[2][0][0]-grid->x[3][0][0]);
				grid->c3b[i][j]=-1.*(grid->c3a[i][j]-1.);
			}
			if (i==1)
			{
				grid->c3a[i][j]=1.+(grid->x[grid->nxm1][0][0]-grid->x[grid->nxm2][0][0])/(grid->x[grid->nxm3][0][0]-grid->x[grid->nxm4][0][0]);
				grid->c3b[i][j]=-1.*(grid->c3a[i][j]-1.);
			}
			if (i==2)
			{
				grid->c3a[i][j]=1.+(grid->y[0][0][0]-grid->y[0][1][0])/(grid->y[0][2][0]-grid->y[0][3][0]);
				grid->c3b[i][j]=-1.*(grid->c3a[i][j]-1.);
			}
			if (i==3)
			{
				grid->c3a[i][j]=1.+(grid->y[0][grid->nym1][0]-grid->y[0][grid->nym2][0])/(grid->y[0][grid->nym3][0]-grid->y[0][grid->nym4][0]);
				grid->c3b[i][j]=-1.*(grid->c3a[i][j]-1.);
			}
			if (i==4)
			{
				grid->c3a[i][j]=1.+(grid->z[0][0][0]-grid->z[0][0][1])/(grid->z[0][0][2]-grid->z[0][0][3]);
				grid->c3b[i][j]=-1.*(grid->c3a[i][j]-1.);
			}
			if (i==5)
			{
				grid->c3a[i][j]=1.+(grid->z[0][0][grid->nzm1]-grid->z[0][0][grid->nzm2])/(grid->z[0][0][grid->nzm3]-grid->z[0][0][grid->nzm4]);
				grid->c3b[i][j]=-1.*(grid->c3a[i][j]-1.);
			}
		}
	}
//	Initialize all values
		if (grid->verb) printf(" --Initializing Arrays\n");
		mhd->md=md0;
		mhd->mi=mi0;
		mhd->me=mi0/1860.;
		mhd->mn=mn0;
		mhd->e=0.72;
		rhoe0=mhd->me*(zi0*rhoi0/mi0-zd0*rho0);
		mhd->ioniz=ioniz0;
		mhd->recom=recom0;
		if (rhoe0 < 0.0)
		{
			printf("---ERROR:  Negative Electron Density---\n");
			printf("---        Check \'config.dat\'\n");
			noutic=0;
			return;	
		}
		p0=rho0*1.0*(gamma0-1.);
		pi0=rhoi0*1.0*(gammai0-1.);
		pe0=rhoe0*1.0*(gammae0-1.);
		pn0=rhon0*1.0*(gamman0-1.);
		sx0=rho0*vx0;
		sy0=rho0*vy0;
		sz0=rho0*vz0;
		six0=rhoi0*vix0;
		siy0=rhoi0*viy0;
		siz0=rhoi0*viz0;
		snx0=rhon0*vnx0;
		sny0=rhon0*vny0;
		snz0=rhon0*vnz0;
		for(i=0;i<grid->nx;i++)
		{
			for(j=0;j<grid->ny;j++)
			{
				for(k=0;k<grid->nz;k++)
				{
					mhd->rho[i][j][k]=rho0;
					mhd->rhoi[i][j][k]=rhoi0;
					mhd->rhoe[i][j][k]=rhoe0;
					mhd->rhon[i][j][k]=rhon0;
					mhd->sx[i][j][k]=sx0;
					mhd->sy[i][j][k]=sy0;
					mhd->sz[i][j][k]=sz0;
					mhd->six[i][j][k]=six0;
					mhd->siy[i][j][k]=siy0;
					mhd->siz[i][j][k]=siz0;
					mhd->sex[i][j][k]=0.0;
					mhd->sey[i][j][k]=0.0;
					mhd->sez[i][j][k]=0.0;
					mhd->snx[i][j][k]=snx0;
					mhd->sny[i][j][k]=sny0;
					mhd->snz[i][j][k]=snz0;
					mhd->zd[i][j][k]=zd0;
					mhd->zi[i][j][k]=zi0;
					mhd->bx[i][j][k]=bx0;
					mhd->by[i][j][k]=by0;
					mhd->bz[i][j][k]=bz0;
					mhd->ex[i][j][k]=0.0;
					mhd->ey[i][j][k]=0.0;
					mhd->ez[i][j][k]=0.0;
					mhd->jx[i][j][k]=0.0;
					mhd->jy[i][j][k]=0.0;
					mhd->jz[i][j][k]=0.0;
					mhd->gamma[i][j][k]=gamma0;
					mhd->gammai[i][j][k]=gammai0;
					mhd->gammae[i][j][k]=gammae0;
					mhd->gamman[i][j][k]=gamman0;
					mhd->p[i][j][k]=p0;
					mhd->pi[i][j][k]=pi0;
					mhd->pe[i][j][k]=pe0;
					mhd->pn[i][j][k]=pn0;
					mhd->nuid[i][j][k]=nuid0;
					mhd->nuie[i][j][k]=nuie0;
					mhd->nude[i][j][k]=nude0;
					mhd->nudn[i][j][k]=nudn0;
					mhd->nuin[i][j][k]=nuin0;
					mhd->nuen[i][j][k]=nuen0;
					mhd->gravx[i][j][k]=0.0;
					mhd->gravy[i][j][k]=0.0;
					mhd->gravz[i][j][k]=0.0;
				}
			}
		}
		grid->kdo=0;
		grid->kio=0;
//	Initial Conditions Block
	if (grid->verb) printf(" --Loading Initial Configuration\n");
//-----------------------Kelvin-Helmholtz----------------------------
//      rho0  : Solar Wind dust (heavy-ion) density
//      rhoi0 : Solar Wind ion density
//      vx0   : Delta-V across boundary
//      vz0   : Amplitude of perturbation
	double d=1.0;
	double offset=5.;
	double deltanh=0.9;
	double deltani=91.;
	double deltarhon=1.;
	double deltaby=0.65;
	double wavenumber=2.*M_PI/5.;
	double kbt=0.67;
        double shift1=grid->xmax/2.;
        double profile=0;
        double sech2p,sech2m,tanhp,tanhm;
        
        
	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				tanhp=(1+tanh((grid->x[i][j][k]+shift1)/d));
				tanhm=(1+tanh((-grid->x[i][j][k]+shift1)/d));
				//Density
                                profile=0.25*tanhp*tanhm;
				mhd->rho[i][j][k]=rho0+deltanh*profile*mhd->md;
				mhd->rhon[i][j][k]=rhon0+deltarhon*profile*mhd->mn;
                                profile=1-profile;
				mhd->rhoi[i][j][k]=rhoi0+deltani*profile*mhd->mi;
				mhd->rhoe[i][j][k]=mhd->me*(mhd->zi[i][j][k]*mhd->rhoi[i][j][k]/mhd->mi-mhd->zd[i][j][k]*mhd->rho[i][j][k]/mhd->md);
				//Momentums (perturbation)
				profile=-0.5+0.25*tanhp*tanhm;
				mhd->sz[i][j][k]  = mhd->rho[i][j][k]  * vz0  * profile;
				mhd->siz[i][j][k] = mhd->rhoi[i][j][k] * viz0 * profile;
				mhd->snz[i][j][k] = mhd->rhon[i][j][k] * vnz0 * profile;
				profile=sin(grid->z[i][j][k]*wavenumber)/cosh((grid->x[i][j][k]-shift1)/d)/cosh((grid->x[i][j][k]-shift1)/d);
				mhd->sx[i][j][k]  = mhd->rho[i][j][k]  * vx0  * profile;
				mhd->six[i][j][k] = mhd->rhoi[i][j][k] * vix0 * profile;
				mhd->snx[i][j][k] = mhd->rhon[i][j][k] * vnx0 * profile;
				profile=sin(grid->z[i][j][k]*wavenumber)/cosh((grid->x[i][j][k]+shift1)/d)/cosh((grid->x[i][j][k]+shift1)/d);
				mhd->sx[i][j][k]  = mhd->sx[i][j][k]  + mhd->rho[i][j][k]  * vx0  * profile;
				mhd->six[i][j][k] = mhd->six[i][j][k] + mhd->rhoi[i][j][k] * vix0 * profile;
				mhd->snx[i][j][k] = mhd->snx[i][j][k] + mhd->rhon[i][j][k] * vnx0 * profile;
				//Pressures (p=n0*kbt)
				mhd->p[i][j][k]=rho0*kbt;
				mhd->pi[i][j][k]=rhoi0*kbt/mhd->mi;
				mhd->pe[i][j][k]=rhoe0*kbt/mhd->me;
                                mhd->pn[i][j][k]=rhon0*kbt/mhd->mn;
                                // Magnetic Field (Constant)
			}
		}
	}
//------  Save Magnetic Field for antisymmetric BC
	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				grid->bx0[i][j][k]=mhd->bx[i][j][k];
				grid->by0[i][j][k]=mhd->by[i][j][k];
				grid->bz0[i][j][k]=mhd->bz[i][j][k];
			}
		}
	}
//	End Initial Conditions block
//	Compute Non-Integrated Arrays
	if (grid->verb) printf(" --Computing Non-Integrated Arrays\n");
	for(i=1;i<grid->nxm1;i++)
	{
		for(j=1;j<grid->nym1;j++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				if (!grid->econt)
				{ 
					mhd->rhoe[i][j][k]=mhd->me*(mhd->zi[i][j][k]*mhd->rhoi[i][j][k]/mhd->mi-mhd->zd[i][j][k]*mhd->rho[i][j][k]/mhd->md);
				}
				else if ((i==1) && (j==1) && (k==1))
				{
					if (grid->verb) printf(" --Electron Continutity On\n");
				}
				if (mhd->rhoe[i][j][k] < 0.0)
				{
					printf("---ERROR: Negative Electron Density---\n");
					printf("          Check Initial Conditions Block\n");
					noutic=0;
					return;
				}
				mhd->jx[i][j][k]=((mhd->bz[i][j+1][k]-mhd->bz[i][j-1][k])*grid->dify[i][j][k]-(mhd->by[i][j][k+1]-mhd->by[i][j][k-1])*grid->difz[i][j][k]);
				mhd->jy[i][j][k]=((mhd->bx[i][j][k+1]-mhd->bx[i][j][k-1])*grid->difz[i][j][k]-(mhd->bz[i+1][j][k]-mhd->bz[i-1][j][k])*grid->difx[i][j][k]);
				mhd->jz[i][j][k]=((mhd->by[i+1][j][k]-mhd->by[i-1][j][k])*grid->difx[i][j][k]-(mhd->bx[i][j+1][k]-mhd->bx[i][j-1][k])*grid->dify[i][j][k]);
				mhd->sex[i][j][k]=(mhd->me/mhd->mi)*mhd->zi[i][j][k]*mhd->six[i][j][k]-(mhd->me/mhd->md)*mhd->zd[i][j][k]*mhd->sx[i][j][k]-mhd->me*mhd->jx[i][j][k]/mhd->e;
				mhd->sey[i][j][k]=(mhd->me/mhd->mi)*mhd->zi[i][j][k]*mhd->siy[i][j][k]-(mhd->me/mhd->md)*mhd->zd[i][j][k]*mhd->sy[i][j][k]-mhd->me*mhd->jy[i][j][k]/mhd->e;
				mhd->sez[i][j][k]=(mhd->me/mhd->mi)*mhd->zi[i][j][k]*mhd->siz[i][j][k]-(mhd->me/mhd->md)*mhd->zd[i][j][k]*mhd->sz[i][j][k]-mhd->me*mhd->jz[i][j][k]/mhd->e;
			}
		}
	}
	for(i=1;i<grid->nxm1;i++)
	{
		for(j=1;j<grid->nym1;j++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				mhd->ex[i][j][k]= mhd->me*(-1.*(grid->difx[i][j][k]*(mhd->pe[i+1][j][k]-mhd->pe[i-1][j][k]))
                                                           -mhd->e*(mhd->sey[i][j][k]*mhd->bz[i][j][k]-mhd->sez[i][j][k]*mhd->by[i][j][k])/mhd->me)/mhd->rhoe[i][j][k]
                                                 +mhd->me/mhd->rhoe[i][j][k]*( mhd->rhoe[i][j][k]*mhd->gravx[i][j][k]
                                                                              -mhd->rho[i][j][k]*mhd->nude[i][j][k]*(mhd->sex[i][j][k]/mhd->rhoe[i][j][k]-mhd->sx[i][j][k]/mhd->rho[i][j][k])
                                                                              -mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*(mhd->sex[i][j][k]/mhd->rhoe[i][j][k]-mhd->six[i][j][k]/mhd->rhoi[i][j][k])
                                                                              -mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*(mhd->sex[i][j][k]/mhd->rhoe[i][j][k]-mhd->snx[i][j][k]/mhd->rhon[i][j][k])
                                                                              +mhd->me/(mhd->mi+mhd->me)*( mhd->ioniz*mhd->snx[i][j][k]
                                                                                                          -mhd->recom*mhd->rhoi[i][j][k]*mhd->sex[i][j][k]/mhd->me))/mhd->e;
				mhd->ey[i][j][k]=mhd->me*(-1.*(grid->dify[i][j][k]*(mhd->pe[i][j+1][k]-mhd->pe[i][j-1][k]))-mhd->e*(mhd->sez[i][j][k]*mhd->bx[i][j][k]-mhd->sex[i][j][k]*mhd->bz[i][j][k])/mhd->me)/mhd->rhoe[i][j][k]+mhd->me/mhd->rhoe[i][j][k]*(mhd->rhoe[i][j][k]*mhd->gravy[i][j][k]-mhd->rho[i][j][k]*mhd->nude[i][j][k]*(mhd->sey[i][j][k]/mhd->rhoe[i][j][k]-mhd->sy[i][j][k]/mhd->rho[i][j][k])-mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*(mhd->sey[i][j][k]/mhd->rhoe[i][j][k]-mhd->siy[i][j][k]/mhd->rhoi[i][j][k])-mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*(mhd->sey[i][j][k]/mhd->rhoe[i][j][k]-mhd->sny[i][j][k]/mhd->rhon[i][j][k])+mhd->me/(mhd->mi+mhd->me)*(mhd->ioniz*mhd->sny[i][j][k]-mhd->recom*mhd->rhoi[i][j][k]*mhd->sey[i][j][k]/mhd->me))/mhd->e;
				mhd->ez[i][j][k]=mhd->me*(-1.*(grid->difz[i][j][k]*(mhd->pe[i][j][k+1]-mhd->pe[i][j][k-1]))-mhd->e*(mhd->sex[i][j][k]*mhd->by[i][j][k]-mhd->sey[i][j][k]*mhd->bx[i][j][k])/mhd->me)/mhd->rhoe[i][j][k]+mhd->me/mhd->rhoe[i][j][k]*(mhd->rhoe[i][j][k]*mhd->gravz[i][j][k]-mhd->rho[i][j][k]*mhd->nude[i][j][k]*(mhd->sez[i][j][k]/mhd->rhoe[i][j][k]-mhd->sz[i][j][k]/mhd->rho[i][j][k])-mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*(mhd->sez[i][j][k]/mhd->rhoe[i][j][k]-mhd->siz[i][j][k]/mhd->rhoi[i][j][k])-mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*(mhd->sez[i][j][k]/mhd->rhoe[i][j][k]-mhd->snz[i][j][k]/mhd->rhon[i][j][k])+mhd->me/(mhd->mi+mhd->me)*(mhd->ioniz*mhd->snz[i][j][k]-mhd->recom*mhd->rhoi[i][j][k]*mhd->sez[i][j][k]/mhd->me))/mhd->e;
			}
		}
	}
}
//*****************************************************************************
/*
	Function:	makegrid
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		9/18/08
	Inputs:		struct grid,xeq,yeq,zeq,dxmax,dymax,dzmax
	Outputs:	none
	Purpose:	This function creates the grids for
			the simulation.
*/
void makegrid(GRID *grid,MHD *mhd,int xeq,int yeq,int zeq,int ldxmin,int ldymin,int ldzmin,double dxmin,double dymin,double dzmin)
{
	int 	i,j,k,ix0;
	double	kx,beta,a1,a2,xanf,w,w1,w2;
	double  sxmax,symax,szmax;

//	X-Axis
	kx=(grid->xmax - grid->xmin)/(double)(grid->nxm3);
	ix0=1;
	beta=M_PI/(grid->xmax-grid->xmin);
	xanf=grid->xmin;
	if ((ldxmin == 0) && (xeq == 0)) xanf=grid->xmin;
	if ((ldxmin == 1) && (xeq == 0))
	{
		xanf=(grid->xmax+grid->xmin)/2.0;
		ix0=(grid->nx)/2.0;
		beta=2.*M_PI/(grid->xmax-grid->xmin);
	}
	if ((ldxmin ==2) && (xeq ==0))
	{
		xanf=grid->xmax;
		ix0=grid->nx-2;
	}
	a1=0.0;
	a2=0.0;
	if (xeq == 0)
	{
		a1=4./3.*(dxmin-kx)/kx/beta;
		a2=(kx-dxmin)/6./kx/beta;
	}
	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				w=kx*(i-ix0);
				grid->x[i][j][k]=w+a1*sin(beta*w)+a2*sin(2.*beta*w);
				grid->difx[i][j][k]=.5/kx/(1.+a1*beta*cos(beta*w)+2.*a2*beta*cos(2.*beta*w));
				w1=w+.5*kx;
				w2=w-.5*kx;
				grid->ddifpx[i][j][k]=2.*grid->difx[i][j][k]/kx/(1.+a1*beta*cos(beta*w1)+2.*a2*beta*cos(2.*beta*w1));
				grid->ddifmx[i][j][k]=2.*grid->difx[i][j][k]/kx/(1.+a1*beta*cos(beta*w2)+2.*a2*beta*cos(2.*beta*w2));
				grid->ddifx[i][j][k]=.5*(grid->ddifpx[i][j][k]+grid->ddifmx[i][j][k]);
			}
		}
	}
	for(k=0;k<grid->nz;k++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(i=1;i<grid->nxm1;i++)
			{
				grid->meanpx[i][j][k]=(grid->x[i+1][j][k]-grid->x[i][j][k])/(grid->x[i+1][j][k]-grid->x[i-1][j][k]);
				grid->meanmx[i][j][k]=(grid->x[i][j][k]-grid->x[i-1][j][k])/(grid->x[i+1][j][k]-grid->x[i-1][j][k]);
			}
			grid->meanmx[0][j][k]=grid->meanpx[2][j][k];
			grid->meanpx[0][j][k]=grid->meanmx[2][j][k];
			grid->meanmx[grid->nx-1][j][k]=grid->meanpx[grid->nx-3][j][k];
			grid->meanpx[grid->nx-1][j][k]=grid->meanmx[grid->nx-3][j][k];
		}
	}
	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				grid->x[i][j][k]=grid->x[i][j][k]+xanf;
			}
		}
	}
//     Y-Axis
	kx=(grid->ymax - grid->ymin)/(double)(grid->nym3);
	ix0=1;
	beta=M_PI/(grid->ymax-grid->ymin);
	xanf=grid->ymin;
	if ((ldymin == 0) && (yeq == 0)) xanf=grid->ymin;
	if ((ldymin == 1) && (yeq == 0))
	{
		xanf=(grid->ymax+grid->ymin)/2.0;
		ix0=grid->ny/2.0;
		beta=2.*M_PI/(grid->ymax-grid->ymin);
	}
	if ((ldymin ==2) && (yeq == 0))
	{
		xanf=grid->ymax;
		ix0=grid->nym2;
	}
	a1=0.0;
	a2=0.0;
	if (yeq == 0)
	{
		a1=4./3.*(dymin-kx)/kx/beta;
		a2=(kx-dymin)/6./kx/beta;
	}
	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				w=kx*(j-ix0);
				grid->y[i][j][k]=w+a1*sin(beta*w)+a2*sin(2.*beta*w);
				grid->dify[i][j][k]=.5/kx/(1.+a1*beta*cos(beta*w)+2.*a2*beta*cos(2.*beta*w));
				w1=w+.5*kx;
				w2=w-.5*kx;
				grid->ddifpy[i][j][k]=2.*grid->dify[i][j][k]/kx/(1.+a1*beta*cos(beta*w1)+2.*a2*beta*cos(2.*beta*w1));
				grid->ddifmy[i][j][k]=2.*grid->dify[i][j][k]/kx/(1.+a1*beta*cos(beta*w2)+2.*a2*beta*cos(2.*beta*w2));
				grid->ddify[i][j][k]=.5*(grid->ddifpy[i][j][k]+grid->ddifmy[i][j][k]);
			}
		}
	}
	for(k=0;k<grid->nz;k++)
	{
		for(i=0;i<grid->nx;i++)
		{
			for(j=1;j<grid->nym1;j++)
			{
				grid->meanpy[i][j][k]=(grid->y[i][j+1][k]-grid->y[i][j][k])/(grid->y[i][j+1][k]-grid->y[i][j-1][k]);
				grid->meanmy[i][j][k]=(grid->y[i][j][k]-grid->y[i][j-1][k])/(grid->y[i][j+1][k]-grid->y[i][j-1][k]);
			}
			grid->meanmy[i][0][k]=grid->meanpy[i][2][k];
			grid->meanpy[i][0][k]=grid->meanmy[i][2][k];
			grid->meanmy[i][grid->nym1][k]=grid->meanpy[i][grid->nym3][k];
			grid->meanpy[i][grid->nym1][k]=grid->meanmy[i][grid->nym3][k];
		
}
	}
	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				grid->y[i][j][k]=grid->y[i][j][k]+xanf;
			}
		}
	}
//    Z-Axis
	kx=(grid->zmax - grid->zmin)/(double)(grid->nzm3);
	ix0=1;
	beta=M_PI/(grid->zmax-grid->zmin);
	xanf=grid->zmin;
	if ((ldzmin == 0) && (zeq == 0)) xanf=grid->zmin;
	if ((ldzmin == 1) && (zeq == 0))
	{
		xanf=(grid->zmax+grid->zmin)/2.0;
		ix0=grid->nz/2.0;
		beta=2.*M_PI/(grid->zmax-grid->zmin);
	}
	if ((ldzmin ==3) && (zeq == 0))
	{
		xanf=grid->zmax;
		ix0=grid->nzm2;
	}
	a1=0.0;
	a2=0.0;
	if (zeq == 0)
	{
		a1=4./3.*(dzmin-kx)/kx/beta;
		a2=(kx-dzmin)/6./kx/beta;
	}
	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				w=kx*(k-ix0);
				grid->z[i][j][k]=w+a1*sin(beta*w)+a2*sin(2.*beta*w);
				grid->difz[i][j][k]=.5/kx/(1.+a1*beta*cos(beta*w)+2.*a2*beta*cos(2.*beta*w));
				w1=w+.5*kx;
				w2=w-.5*kx;
				grid->ddifpz[i][j][k]=2.*grid->difz[i][j][k]/kx/(1.+a1*beta*cos(beta*w1)+2.*a2*beta*cos(2.*beta*w1));
				grid->ddifmz[i][j][k]=2.*grid->difz[i][j][k]/kx/(1.+a1*beta*cos(beta*w2)+2.*a2*beta*cos(2.*beta*w2));
				grid->ddifz[i][j][k]=.5*(grid->ddifpz[i][j][k]+grid->ddifmz[i][j][k]);
			}
		}
	}
	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				grid->meanpz[i][j][k]=(grid->z[i][j][k+1]-grid->z[i][j][k])/(grid->z[i][j][k+1]-grid->z[i][j][k-1]);
				grid->meanmz[i][j][k]=(grid->z[i][j][k]-grid->z[i][j][k-1])/(grid->z[i][j][k+1]-grid->z[i][j][k-1]);
			}
			grid->meanmz[i][j][0]=grid->meanpz[i][j][2];
			grid->meanpz[i][j][0]=grid->meanmz[i][j][2];
			grid->meanmz[i][j][grid->nzm1]=grid->meanpz[i][j][grid->nzm3];
			grid->meanpz[i][j][grid->nzm1]=grid->meanmz[i][j][grid->nzm3];
		}
	}
	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				grid->z[i][j][k]=grid->z[i][j][k]+xanf;
			}
		}
	}
//	Make FTCS smoothing grids
	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				grid->ssx[i][j][k]=mhd->visx*grid->ddifx[i][j][k];
				grid->ssy[i][j][k]=mhd->visy*grid->ddify[i][j][k];
				grid->ssz[i][j][k]=mhd->visz*grid->ddifz[i][j][k];
				grid->s[i][j][k]=1.0-2.*(grid->ssx[i][j][k]+grid->ssy[i][j][k]+grid->ssz[i][j][k]);
			}
		}
	}
//      Now output some stuff
	if (grid->verb)
	{
		printf("      Grid Values\n");
		printf("      -----------\n");
		printf("        x     = [ %+5.4f, %+5.4f ]\n",grid->x[0][0][0],grid->x[grid->nxm1][0][0]);
		printf("        difx  = [ %+5.4f, %+5.4f ]\n",min3d(grid->difx,grid->nx,grid->ny,grid->nz),max3d(grid->difx,grid->nx,grid->ny,grid->nz));
		printf("        ddifx = [ %+5.4f, %+5.4f ]\n",min3d(grid->ddifx,grid->nx,grid->ny,grid->nz),max3d(grid->ddifx,grid->nx,grid->ny,grid->nz));
		printf("        y     = [ %+5.4f, %+5.4f ]\n",grid->y[0][0][0],grid->y[0][grid->nym1][0]);
		printf("        dify  = [ %+5.4f, %+5.4f ]\n",min3d(grid->dify,grid->nx,grid->ny,grid->nz),max3d(grid->dify,grid->nx,grid->ny,grid->nz));
		printf("        ddify = [ %+5.4f, %+5.4f ]\n",min3d(grid->ddify,grid->nx,grid->ny,grid->nz),max3d(grid->ddify,grid->nx,grid->ny,grid->nz));
		printf("        z     = [ %+5.4f, %+5.4f ]\n",grid->z[0][0][0],grid->z[0][0][grid->nzm1]);
		printf("        difz  = [ %+5.4f, %+5.4f ]\n",min3d(grid->difz,grid->nx,grid->ny,grid->nz),max3d(grid->difz,grid->nx,grid->ny,grid->nz));
		printf("        ddifz = [ %+5.4f, %+5.4f ]\n",min3d(grid->ddifz,grid->nx,grid->ny,grid->nz),max3d(grid->ddifz,grid->nx,grid->ny,grid->nz));
		printf("      Stability\n");
		printf("      ---------\n");
		printf("        Cx     = [ %5.4f, %5.4f ]   (Cx=dt/dx)\n",2.*grid->dt*min3d(grid->difx,grid->nx,grid->ny,grid->nz),
			                                     2.*grid->dt*max3d(grid->difx,grid->nx,grid->ny,grid->nz));
		printf("        Cy     = [ %5.4f, %5.4f ]\n",2.*grid->dt*min3d(grid->dify,grid->nx,grid->ny,grid->nz),
			                                     2.*grid->dt*max3d(grid->dify,grid->nx,grid->ny,grid->nz));
		printf("        Cz     = [ %5.4f, %5.4f ]\n",2.*grid->dt*min3d(grid->difz,grid->nx,grid->ny,grid->nz),
			                                     2.*grid->dt*max3d(grid->difz,grid->nx,grid->ny,grid->nz));
		printf("        Sx     = [ %5.4f, %5.4f ]   (Sx=alpha/dx^2)\n",mhd->visx*min3d(grid->ddifx,grid->nx,grid->ny,grid->nz),
			                                                       mhd->visx*max3d(grid->ddifx,grid->nx,grid->ny,grid->nz));
		printf("        Sy     = [ %5.4f, %5.4f ]\n",mhd->visy*min3d(grid->ddify,grid->nx,grid->ny,grid->nz),
			                                     mhd->visy*max3d(grid->ddify,grid->nx,grid->ny,grid->nz));
		printf("        Sz     = [ %5.4f, %5.4f ]\n",mhd->visz*min3d(grid->ddifz,grid->nx,grid->ny,grid->nz),
			                                     mhd->visz*max3d(grid->ddifz,grid->nx,grid->ny,grid->nz));
		sxmax=mhd->visx*max3d(grid->ddifx,grid->nx,grid->ny,grid->nz);
		symax=mhd->visy*max3d(grid->ddify,grid->nx,grid->ny,grid->nz);
		szmax=mhd->visz*max3d(grid->ddifz,grid->nx,grid->ny,grid->nz);
		if ((sxmax+symax+szmax) > (1./6.))
		{
			printf("=====  WARNING:  0 < (Sx+Sy+Sz) <= 1/6  =====\n");
			printf(" Sx+Sy+Sz=%5.4f\n",sxmax+symax+szmax);
			printf(" Increase Gridsize or Decrease visx,visy,visz\n");
			printf("=============================================\n");
			exit(-1);
		}
	}

}
//*****************************************************************************
/*
	Function:	bcorg_nmhdust
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		9/26/08
	Inputs:		MHD mhd,GRID grid
	Outputs:	none
	Purpose:	Organizes Application of Boundary Conditions
			The general method used for applying boundary conditions
			is
			f(i+1) = c1*f(i-1) + c2*f(i-3) + perio*f(i-3)
			for periodicity we simply set
			f(0)   = f(N-3)
			f(N-1) = f(2)
			Now updates jx,jy,jz,sex,sey,sez for
			boundaries using six,siy,siz parameters
			perx = 2 now executes time dependant BC Blocks
			Current and Electron momentum now handled after
			integrated quantities

*/
void	bcorg_nmhdust(MHD *mhd,GRID *grid,double *time)
{
	int i,j,k,l,m,n,tid;
	int ip1,jp1,kp1;
	int im1,jm1,km1;
	double jfac1,jfac2,jfac3,jfac4,jfac5,jfac6;
        double freq,wavenum,amp,at,third;
	double meinv,miinv,mdinv,memi,memd,stime;

//	Create some helper variables
	meinv=1./mhd->me;
	miinv=1./mhd->mi;
        mdinv=1./mhd->md;
	memi=mhd->me/mhd->mi;
	memd=mhd->me/mhd->md;
	third=1./3.;


	#pragma omp parallel default(shared) private(i,j,k,stime,tid,ip1,im1,jp1,jm1,kp1,km1,jfac1,jfac2,jfac3,jfac4,jfac5,jfac6)
	{		
//		tid=omp_get_thread_num();
//		if (tid == 0)
//		{
//			stime=omp_get_wtime();
//			printf("----- BCORG");
//		}
//      X-BC
	if (grid->perx == 1)
	{
//		X-min
		#pragma omp for nowait
		for(j=1;j<grid->nym1;j++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				mhd-> rho[0][j][k]=mhd-> rho[grid->nxm3][j][k];
				mhd->rhoi[0][j][k]=mhd->rhoi[grid->nxm3][j][k];
				mhd->rhoe[0][j][k]=mhd->rhoe[grid->nxm3][j][k];
				mhd->rhon[0][j][k]=mhd->rhon[grid->nxm3][j][k];
				mhd->  sx[0][j][k]=mhd->  sx[grid->nxm3][j][k];
				mhd->  sy[0][j][k]=mhd->  sy[grid->nxm3][j][k];
				mhd->  sz[0][j][k]=mhd->  sz[grid->nxm3][j][k];
				mhd-> six[0][j][k]=mhd-> six[grid->nxm3][j][k];
				mhd-> siy[0][j][k]=mhd-> siy[grid->nxm3][j][k];
				mhd-> siz[0][j][k]=mhd-> siz[grid->nxm3][j][k];
				mhd-> snx[0][j][k]=mhd-> snx[grid->nxm3][j][k];
				mhd-> sny[0][j][k]=mhd-> sny[grid->nxm3][j][k];
				mhd-> snz[0][j][k]=mhd-> snz[grid->nxm3][j][k];
				mhd->  bx[0][j][k]=mhd->  bx[grid->nxm3][j][k];
				mhd->  by[0][j][k]=mhd->  by[grid->nxm3][j][k];
				mhd->  bz[0][j][k]=mhd->  bz[grid->nxm3][j][k];
				mhd->   p[0][j][k]=mhd->   p[grid->nxm3][j][k];
				mhd->  pi[0][j][k]=mhd->  pi[grid->nxm3][j][k];
				mhd->  pe[0][j][k]=mhd->  pe[grid->nxm3][j][k];
				mhd->  pn[0][j][k]=mhd->  pn[grid->nxm3][j][k];
			}
		}
//		X-max
		#pragma omp for
		for(j=1;j<grid->nym1;j++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				mhd-> rho[grid->nxm1][j][k]=mhd-> rho[2][j][k];
				mhd->rhoi[grid->nxm1][j][k]=mhd->rhoi[2][j][k];
				mhd->rhoe[grid->nxm1][j][k]=mhd->rhoe[2][j][k];
				mhd->rhon[grid->nxm1][j][k]=mhd->rhon[2][j][k];
				mhd->  sx[grid->nxm1][j][k]=mhd->  sx[2][j][k];
				mhd->  sy[grid->nxm1][j][k]=mhd->  sy[2][j][k];
				mhd->  sz[grid->nxm1][j][k]=mhd->  sz[2][j][k];
				mhd-> six[grid->nxm1][j][k]=mhd-> six[2][j][k];
				mhd-> siy[grid->nxm1][j][k]=mhd-> siy[2][j][k];
				mhd-> siz[grid->nxm1][j][k]=mhd-> siz[2][j][k];
				mhd-> snx[grid->nxm1][j][k]=mhd-> snx[2][j][k];
				mhd-> sny[grid->nxm1][j][k]=mhd-> sny[2][j][k];
				mhd-> snz[grid->nxm1][j][k]=mhd-> snz[2][j][k];
				mhd->  bx[grid->nxm1][j][k]=mhd->  bx[2][j][k];
				mhd->  by[grid->nxm1][j][k]=mhd->  by[2][j][k];
				mhd->  bz[grid->nxm1][j][k]=mhd->  bz[2][j][k];
				mhd->   p[grid->nxm1][j][k]=mhd->   p[2][j][k];
				mhd->  pi[grid->nxm1][j][k]=mhd->  pi[2][j][k];
				mhd->  pe[grid->nxm1][j][k]=mhd->  pe[2][j][k];
				mhd->  pn[grid->nxm1][j][k]=mhd->  pn[2][j][k];
			}
		}
	}
	else if (grid->perx == 0)
	{
		#pragma omp for nowait
		for(j=1;j<grid->nym1;j++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				//X-Min
				mhd->rho[0][j][k]=grid->c1[0][0]*mhd->rho[2][j][k]+grid->c2[0][0]*mhd->rho[4][j][k]+grid->d[0][0]*(grid->c3a[0][0]*mhd->rho[1][j][k]+grid->c3b[0][0]*mhd->rho[3][j][k]);
				mhd->rhoi[0][j][k]=grid->c1[0][1]*mhd->rhoi[2][j][k]+grid->c2[0][1]*mhd->rhoi[4][j][k]+grid->d[0][1]*(grid->c3a[0][1]*mhd->rhoi[1][j][k]+grid->c3b[0][1]*mhd->rhoi[3][j][k]);
				mhd->rhoe[0][j][k]=mhd->me*(mhd->zi[0][j][k]*mhd->rhoi[0][j][k]*miinv-mhd->zd[0][j][k]*mhd->rho[0][j][k]*mdinv);
				mhd->rhon[0][j][k]=grid->c1[0][3]*mhd->rhon[2][j][k]+grid->c2[0][3]*mhd->rhon[4][j][k]+grid->d[0][3]*(grid->c3a[0][3]*mhd->rhon[1][j][k]+grid->c3b[0][3]*mhd->rhon[3][j][k]);
				mhd->p[0][j][k]=grid->c1[0][4]*mhd->p[2][j][k]+grid->c2[0][4]*mhd->p[4][j][k]+grid->d[0][4]*(grid->c3a[0][4]*mhd->p[1][j][k]+grid->c3b[0][4]*mhd->p[3][j][k]);
				mhd->pi[0][j][k]=grid->c1[0][5]*mhd->pi[2][j][k]+grid->c2[0][5]*mhd->pi[5][j][k]+grid->d[0][5]*(grid->c3a[0][5]*mhd->pi[1][j][k]+grid->c3b[0][5]*mhd->pi[3][j][k]);
				mhd->pe[0][j][k]=grid->c1[0][6]*mhd->pe[2][j][k]+grid->c2[0][6]*mhd->pe[4][j][k]+grid->d[0][6]*(grid->c3a[0][6]*mhd->pe[1][j][k]+grid->c3b[0][6]*mhd->pe[3][j][k]);
				mhd->pn[0][j][k]=grid->c1[0][7]*mhd->pn[2][j][k]+grid->c2[0][7]*mhd->pn[4][j][k]+grid->d[0][7]*(grid->c3a[0][7]*mhd->pn[1][j][k]+grid->c3b[0][7]*mhd->pn[3][j][k]);
				mhd->bx[0][j][k]=grid->c1[0][8]*mhd->bx[2][j][k]+grid->c2[0][8]*mhd->bx[4][j][k]+grid->d[0][8]*(grid->c3a[0][8]*mhd->bx[1][j][k]+grid->c3b[0][8]*mhd->bx[3][j][k]);
				mhd->by[0][j][k]=grid->c1[0][9]*mhd->by[2][j][k]+grid->c2[0][9]*mhd->by[4][j][k]+grid->d[0][9]*(grid->c3a[0][9]*mhd->by[1][j][k]+grid->c3b[0][9]*mhd->by[3][j][k]);
				mhd->bz[0][j][k]=grid->c1[0][10]*mhd->bz[2][j][k]+grid->c2[0][10]*mhd->bz[4][j][k]+grid->d[0][10]*(grid->c3a[0][10]*mhd->bz[1][j][k]+grid->c3b[0][10]*mhd->bz[3][j][k]);
				mhd->sx[0][j][k]=grid->c1[0][11]*mhd->sx[2][j][k]+grid->c2[0][11]*mhd->sx[4][j][k]+grid->d[0][11]*(grid->c3a[0][11]*mhd->sx[1][j][k]+grid->c3b[0][11]*mhd->sx[3][j][k]);
				mhd->sy[0][j][k]=grid->c1[0][12]*mhd->sy[2][j][k]+grid->c2[0][12]*mhd->sy[4][j][k]+grid->d[0][12]*(grid->c3a[0][12]*mhd->sy[1][j][k]+grid->c3b[0][12]*mhd->sy[3][j][k]);
				mhd->sz[0][j][k]=grid->c1[0][13]*mhd->sz[2][j][k]+grid->c2[0][13]*mhd->sz[4][j][k]+grid->d[0][13]*(grid->c3a[0][13]*mhd->sz[1][j][k]+grid->c3b[0][13]*mhd->sz[3][j][k]);
				mhd->six[0][j][k]=grid->c1[0][11]*mhd->six[2][j][k]+grid->c2[0][11]*mhd->six[4][j][k]+grid->d[0][11]*(grid->c3a[0][11]*mhd->six[1][j][k]+grid->c3b[0][11]*mhd->six[3][j][k]);
				mhd->siy[0][j][k]=grid->c1[0][12]*mhd->siy[2][j][k]+grid->c2[0][12]*mhd->siy[4][j][k]+grid->d[0][12]*(grid->c3a[0][12]*mhd->siy[1][j][k]+grid->c3b[0][12]*mhd->siy[3][j][k]);
				mhd->siz[0][j][k]=grid->c1[0][13]*mhd->siz[2][j][k]+grid->c2[0][13]*mhd->siz[4][j][k]+grid->d[0][13]*(grid->c3a[0][13]*mhd->siz[1][j][k]+grid->c3b[0][13]*mhd->siz[3][j][k]);
				mhd->snx[0][j][k]=grid->c1[0][11]*mhd->snx[2][j][k]+grid->c2[0][11]*mhd->snx[4][j][k]+grid->d[0][11]*(grid->c3a[0][11]*mhd->snx[1][j][k]+grid->c3b[0][11]*mhd->snx[3][j][k]);
				mhd->sny[0][j][k]=grid->c1[0][12]*mhd->sny[2][j][k]+grid->c2[0][12]*mhd->sny[4][j][k]+grid->d[0][12]*(grid->c3a[0][12]*mhd->sny[1][j][k]+grid->c3b[0][12]*mhd->sny[3][j][k]);
				mhd->snz[0][j][k]=grid->c1[0][13]*mhd->snz[2][j][k]+grid->c2[0][13]*mhd->snz[4][j][k]+grid->d[0][13]*(grid->c3a[0][13]*mhd->snz[1][j][k]+grid->c3b[0][13]*mhd->snz[3][j][k]);
				//Check for k=-1
				if(grid->k[0][8]==-1) mhd->bx[1][j][k]=0.;
				if(grid->k[0][9]==-1) mhd->by[1][j][k]=0.;
				if(grid->k[0][10]==-1) mhd->bz[1][j][k]=0.;
				if(grid->k[0][11]==-1) mhd->sx[1][j][k]=0.;
				if(grid->k[0][12]==-1) mhd->sy[1][j][k]=0.;
				if(grid->k[0][13]==-1) mhd->sz[1][j][k]=0.;
				if(grid->k[0][11]==-1) mhd->six[1][j][k]=0.;
				if(grid->k[0][12]==-1) mhd->siy[1][j][k]=0.;
				if(grid->k[0][13]==-1) mhd->siz[1][j][k]=0.;
				if(grid->k[0][11]==-1) mhd->snx[1][j][k]=0.;
				if(grid->k[0][12]==-1) mhd->sny[1][j][k]=0.;
				if(grid->k[0][13]==-1) mhd->snz[1][j][k]=0.;
				//X-Max
				mhd->rho[grid->nxm1][j][k]=grid->c1[1][0]*mhd->rho[grid->nxm3][j][k]+grid->c2[1][0]*mhd->rho[grid->nx-5][j][k]+grid->d[1][0]*(grid->c3a[1][0]*mhd->rho[grid->nxm2][j][k]+grid->c3b[1][0]*mhd->rho[grid->nxm4][j][k]);
				mhd->rhoi[grid->nxm1][j][k]=grid->c1[1][1]*mhd->rhoi[grid->nxm3][j][k]+grid->c2[1][1]*mhd->rhoi[grid->nx-5][j][k]+grid->d[1][1]*(grid->c3a[1][1]*mhd->rhoi[grid->nxm2][j][k]+grid->c3b[1][1]*mhd->rhoi[grid->nxm4][j][k]);
				mhd->rhoe[grid->nxm1][j][k]=mhd->me*(mhd->zi[grid->nxm1][j][k]*mhd->rhoi[grid->nxm1][j][k]*miinv-mhd->zd[grid->nxm1][j][k]*mhd->rho[grid->nxm1][j][k]*mdinv);
				mhd->rhon[grid->nxm1][j][k]=grid->c1[1][3]*mhd->rhon[grid->nxm3][j][k]+grid->c2[1][3]*mhd->rhon[grid->nx-5][j][k]+grid->d[1][3]*(grid->c3a[1][3]*mhd->rhon[grid->nxm2][j][k]+grid->c3b[1][3]*mhd->rhon[grid->nxm4][j][k]);
				mhd->p[grid->nxm1][j][k]=grid->c1[1][4]*mhd->p[grid->nxm3][j][k]+grid->c2[1][4]*mhd->p[grid->nx-5][j][k]+grid->d[1][4]*(grid->c3a[1][4]*mhd->p[grid->nxm2][j][k]+grid->c3b[1][4]*mhd->p[grid->nxm4][j][k]);
				mhd->pi[grid->nxm1][j][k]=grid->c1[1][5]*mhd->pi[grid->nxm3][j][k]+grid->c2[1][5]*mhd->pi[grid->nx-5][j][k]+grid->d[1][5]*(grid->c3a[1][5]*mhd->pi[grid->nxm2][j][k]+grid->c3b[1][5]*mhd->pi[grid->nxm4][j][k]);
				mhd->pe[grid->nxm1][j][k]=grid->c1[1][6]*mhd->pe[grid->nxm3][j][k]+grid->c2[1][6]*mhd->pe[grid->nx-5][j][k]+grid->d[1][6]*(grid->c3a[1][6]*mhd->pe[grid->nxm2][j][k]+grid->c3b[1][6]*mhd->pe[grid->nxm4][j][k]);
				mhd->pn[grid->nxm1][j][k]=grid->c1[1][7]*mhd->pn[grid->nxm3][j][k]+grid->c2[1][7]*mhd->pn[grid->nx-5][j][k]+grid->d[1][7]*(grid->c3a[1][7]*mhd->pn[grid->nxm2][j][k]+grid->c3b[1][7]*mhd->pn[grid->nxm4][j][k]);
				mhd->bx[grid->nxm1][j][k]=grid->c1[1][8]*mhd->bx[grid->nxm3][j][k]+grid->c2[1][8]*mhd->bx[grid->nx-5][j][k]+grid->d[1][8]*(grid->c3a[1][8]*mhd->bx[grid->nxm2][j][k]+grid->c3b[1][8]*mhd->bx[grid->nxm4][j][k]);
				mhd->by[grid->nxm1][j][k]=grid->c1[1][9]*mhd->by[grid->nxm3][j][k]+grid->c2[1][9]*mhd->by[grid->nx-5][j][k]+grid->d[1][9]*(grid->c3a[1][9]*mhd->by[grid->nxm2][j][k]+grid->c3b[1][9]*mhd->by[grid->nxm4][j][k]);
				mhd->bz[grid->nxm1][j][k]=grid->c1[1][10]*mhd->bz[grid->nxm3][j][k]+grid->c2[1][10]*mhd->bz[grid->nx-5][j][k]+grid->d[1][10]*(grid->c3a[1][10]*mhd->bz[grid->nxm2][j][k]+grid->c3b[1][10]*mhd->bz[grid->nxm4][j][k]);
				mhd->sx[grid->nxm1][j][k]=grid->c1[1][11]*mhd->sx[grid->nxm3][j][k]+grid->c2[1][11]*mhd->sx[grid->nx-5][j][k]+grid->d[1][11]*(grid->c3a[1][11]*mhd->sx[grid->nxm2][j][k]+grid->c3b[1][11]*mhd->sx[grid->nxm4][j][k]);
				mhd->sy[grid->nxm1][j][k]=grid->c1[1][12]*mhd->sy[grid->nxm3][j][k]+grid->c2[1][12]*mhd->sy[grid->nx-5][j][k]+grid->d[1][12]*(grid->c3a[1][12]*mhd->sy[grid->nxm2][j][k]+grid->c3b[1][12]*mhd->sy[grid->nxm4][j][k]);
				mhd->sz[grid->nxm1][j][k]=grid->c1[1][13]*mhd->sz[grid->nxm3][j][k]+grid->c2[1][13]*mhd->sz[grid->nx-5][j][k]+grid->d[1][13]*(grid->c3a[1][13]*mhd->sz[grid->nxm2][j][k]+grid->c3b[1][13]*mhd->sz[grid->nxm4][j][k]);
				mhd->six[grid->nxm1][j][k]=grid->c1[1][11]*mhd->six[grid->nxm3][j][k]+grid->c2[1][11]*mhd->six[grid->nx-5][j][k]+grid->d[1][11]*(grid->c3a[1][11]*mhd->six[grid->nxm2][j][k]+grid->c3b[1][11]*mhd->six[grid->nxm4][j][k]);
				mhd->siy[grid->nxm1][j][k]=grid->c1[1][12]*mhd->siy[grid->nxm3][j][k]+grid->c2[1][12]*mhd->siy[grid->nx-5][j][k]+grid->d[1][12]*(grid->c3a[1][12]*mhd->siy[grid->nxm2][j][k]+grid->c3b[1][12]*mhd->siy[grid->nxm4][j][k]);
				mhd->siz[grid->nxm1][j][k]=grid->c1[1][13]*mhd->siz[grid->nxm3][j][k]+grid->c2[1][13]*mhd->siz[grid->nx-5][j][k]+grid->d[1][13]*(grid->c3a[1][13]*mhd->siz[grid->nxm2][j][k]+grid->c3b[1][13]*mhd->siz[grid->nxm4][j][k]);
				mhd->snx[grid->nxm1][j][k]=grid->c1[1][11]*mhd->snx[grid->nxm3][j][k]+grid->c2[1][11]*mhd->snx[grid->nx-5][j][k]+grid->d[1][11]*(grid->c3a[1][11]*mhd->snx[grid->nxm2][j][k]+grid->c3b[1][11]*mhd->snx[grid->nxm4][j][k]);
				mhd->sny[grid->nxm1][j][k]=grid->c1[1][12]*mhd->sny[grid->nxm3][j][k]+grid->c2[1][12]*mhd->sny[grid->nx-5][j][k]+grid->d[1][12]*(grid->c3a[1][12]*mhd->sny[grid->nxm2][j][k]+grid->c3b[1][12]*mhd->sny[grid->nxm4][j][k]);
				mhd->snz[grid->nxm1][j][k]=grid->c1[1][13]*mhd->snz[grid->nxm3][j][k]+grid->c2[1][13]*mhd->snz[grid->nx-5][j][k]+grid->d[1][13]*(grid->c3a[1][13]*mhd->snz[grid->nxm2][j][k]+grid->c3b[1][13]*mhd->snz[grid->nxm4][j][k]);
				//Check for k=-1
				if(grid->k[1][8]==-1) mhd->bx[grid->nxm2][j][k]=0.;
				if(grid->k[1][9]==-1) mhd->by[grid->nxm2][j][k]=0.;
				if(grid->k[1][10]==-1) mhd->bz[grid->nxm2][j][k]=0.;
				if(grid->k[1][11]==-1) mhd->sx[grid->nxm2][j][k]=0.;
				if(grid->k[1][12]==-1) mhd->sy[grid->nxm2][j][k]=0.;
				if(grid->k[1][13]==-1) mhd->sz[grid->nxm2][j][k]=0.;
				if(grid->k[1][11]==-1) mhd->six[grid->nxm2][j][k]=0.;
				if(grid->k[1][12]==-1) mhd->siy[grid->nxm2][j][k]=0.;
				if(grid->k[1][13]==-1) mhd->siz[grid->nxm2][j][k]=0.;
				if(grid->k[1][11]==-1) mhd->snx[grid->nxm2][j][k]=0.;
				if(grid->k[1][12]==-1) mhd->sny[grid->nxm2][j][k]=0.;
				if(grid->k[1][13]==-1) mhd->snz[grid->nxm2][j][k]=0.;
			}
		}
		#pragma omp barrier
	}
//	X Time Dependant BC's
	else if (grid->perx == 2)
	{
		#pragma omp barrier
	}
//      Y-BC (modified for open BC)
	if (grid->pery == 1)
	{
//		Y-min
		#pragma omp for nowait
		for(i=1;i<grid->nxm1;i++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				mhd-> rho[i][0][k]=mhd-> rho[i][grid->nym3][k];
				mhd->rhoi[i][0][k]=mhd->rhoi[i][grid->nym3][k];
				mhd->rhoe[i][0][k]=mhd->rhoe[i][grid->nym3][k];
				mhd->rhon[i][0][k]=mhd->rhon[i][grid->nym3][k];
				mhd->  sx[i][0][k]=mhd->  sx[i][grid->nym3][k];
				mhd->  sy[i][0][k]=mhd->  sy[i][grid->nym3][k];
				mhd->  sz[i][0][k]=mhd->  sz[i][grid->nym3][k];
				mhd-> six[i][0][k]=mhd-> six[i][grid->nym3][k];
				mhd-> siy[i][0][k]=mhd-> siy[i][grid->nym3][k];
				mhd-> siz[i][0][k]=mhd-> siz[i][grid->nym3][k];
				mhd-> snx[i][0][k]=mhd-> snx[i][grid->nym3][k];
				mhd-> sny[i][0][k]=mhd-> sny[i][grid->nym3][k];
				mhd-> snz[i][0][k]=mhd-> snz[i][grid->nym3][k];
				mhd->  bx[i][0][k]=mhd->  bx[i][grid->nym3][k];
				mhd->  by[i][0][k]=mhd->  by[i][grid->nym3][k];
				mhd->  bz[i][0][k]=mhd->  bz[i][grid->nym3][k];
				mhd->   p[i][0][k]=mhd->   p[i][grid->nym3][k];
				mhd->  pi[i][0][k]=mhd->  pi[i][grid->nym3][k];
				mhd->  pe[i][0][k]=mhd->  pe[i][grid->nym3][k];
				mhd->  pn[i][0][k]=mhd->  pn[i][grid->nym3][k];
			}
		}
//		Y-max
		#pragma omp for
		for(i=1;i<grid->nxm1;i++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				mhd-> rho[i][grid->nym1][k]=mhd-> rho[i][2][k];
				mhd->rhoi[i][grid->nym1][k]=mhd->rhoi[i][2][k];
				mhd->rhoe[i][grid->nym1][k]=mhd->rhoe[i][2][k];
				mhd->rhon[i][grid->nym1][k]=mhd->rhon[i][2][k];
				mhd->  sx[i][grid->nym1][k]=mhd->  sx[i][2][k];
				mhd->  sy[i][grid->nym1][k]=mhd->  sy[i][2][k];
				mhd->  sz[i][grid->nym1][k]=mhd->  sz[i][2][k];
				mhd-> six[i][grid->nym1][k]=mhd-> six[i][2][k];
				mhd-> siy[i][grid->nym1][k]=mhd-> siy[i][2][k];
				mhd-> siz[i][grid->nym1][k]=mhd-> siz[i][2][k];
				mhd-> snx[i][grid->nym1][k]=mhd-> snx[i][2][k];
				mhd-> sny[i][grid->nym1][k]=mhd-> sny[i][2][k];
				mhd-> snz[i][grid->nym1][k]=mhd-> snz[i][2][k];
				mhd->  bx[i][grid->nym1][k]=mhd->  bx[i][2][k];
				mhd->  by[i][grid->nym1][k]=mhd->  by[i][2][k];
				mhd->  bz[i][grid->nym1][k]=mhd->  bz[i][2][k];
				mhd->   p[i][grid->nym1][k]=mhd->   p[i][2][k];
				mhd->  pi[i][grid->nym1][k]=mhd->  pi[i][2][k];
				mhd->  pe[i][grid->nym1][k]=mhd->  pe[i][2][k];
				mhd->  pn[i][grid->nym1][k]=mhd->  pn[i][2][k];
			}
		}
		#pragma omp barrier
	}
	else if (grid->pery == 0)
	{
		#pragma omp for nowait
		for(i=1;i<grid->nxm1;i++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				//Y-Min
				mhd->rho[i][0][k]=grid->c1[2][0]*mhd->rho[i][2][k]+grid->c2[2][0]*mhd->rho[i][4][k]+grid->d[2][0]*(grid->c3a[2][0]*mhd->rho[i][1][k]+grid->c3b[2][0]*mhd->rho[i][3][k]);
				mhd->rhoi[i][0][k]=grid->c1[2][1]*mhd->rhoi[i][2][k]+grid->c2[2][1]*mhd->rhoi[i][4][k]+grid->d[2][1]*(grid->c3a[2][1]*mhd->rhoi[i][1][k]+grid->c3b[2][1]*mhd->rhoi[i][3][k]);
				mhd->rhoe[i][0][k]=mhd->me*(mhd->zi[i][0][k]*mhd->rhoi[i][0][k]*miinv-mhd->zd[i][0][k]*mhd->rho[i][0][k]*mdinv);
				mhd->rhon[i][0][k]=grid->c1[2][3]*mhd->rhon[i][2][k]+grid->c2[2][3]*mhd->rhon[i][4][k]+grid->d[2][3]*(grid->c3a[2][3]*mhd->rhon[i][1][k]+grid->c3b[2][3]*mhd->rhon[i][3][k]);
				mhd->p[i][0][k]=grid->c1[2][4]*mhd->p[i][2][k]+grid->c2[2][4]*mhd->p[i][4][k]+grid->d[2][4]*(grid->c3a[2][4]*mhd->p[i][1][k]+grid->c3b[2][4]*mhd->p[i][3][k]);
				mhd->pi[i][0][k]=grid->c1[2][5]*mhd->pi[i][2][k]+grid->c2[2][5]*mhd->pi[i][4][k]+grid->d[2][5]*(grid->c3a[2][5]*mhd->pi[i][1][k]+grid->c3b[2][5]*mhd->pi[i][3][k]);
				mhd->pe[i][0][k]=grid->c1[2][6]*mhd->pe[i][2][k]+grid->c2[2][6]*mhd->pe[i][4][k]+grid->d[2][6]*(grid->c3a[2][6]*mhd->pe[i][1][k]+grid->c3b[2][6]*mhd->pe[i][3][k]);
				mhd->pn[i][0][k]=grid->c1[2][7]*mhd->pn[i][2][k]+grid->c2[2][7]*mhd->pn[i][4][k]+grid->d[2][7]*(grid->c3a[2][7]*mhd->pn[i][1][k]+grid->c3b[2][7]*mhd->pn[i][3][k]);
				mhd->bx[i][0][k]=grid->c1[2][8]*mhd->bx[i][2][k]+grid->c2[2][8]*mhd->bx[i][4][k]+grid->d[2][8]*(grid->c3a[2][8]*mhd->bx[i][1][k]+grid->c3b[2][8]*mhd->bx[i][3][k]);
				mhd->by[i][0][k]=grid->c1[2][9]*mhd->by[i][2][k]+grid->c2[2][9]*mhd->by[i][4][k]+grid->d[2][9]*(grid->c3a[2][9]*mhd->by[i][1][k]+grid->c3b[2][9]*mhd->by[i][3][k]);
				mhd->bz[i][0][k]=grid->c1[2][10]*mhd->bz[i][2][k]+grid->c2[2][10]*mhd->bz[i][4][k]+grid->d[2][10]*(grid->c3a[2][10]*mhd->bz[i][1][k]+grid->c3b[2][10]*mhd->bz[i][3][k]);
				mhd->sx[i][0][k]=grid->c1[2][11]*mhd->sx[i][2][k]+grid->c2[2][11]*mhd->sx[i][4][k]+grid->d[2][11]*(grid->c3a[2][11]*mhd->sx[i][1][k]+grid->c3b[2][11]*mhd->sx[i][3][k]);
				mhd->sy[i][0][k]=grid->c1[2][12]*mhd->sy[i][2][k]+grid->c2[2][12]*mhd->sy[i][4][k]+grid->d[2][12]*(grid->c3a[2][12]*mhd->sy[i][1][k]+grid->c3b[2][12]*mhd->sy[i][3][k]);
				mhd->sz[i][0][k]=grid->c1[2][13]*mhd->sz[i][2][k]+grid->c2[2][13]*mhd->sz[i][4][k]+grid->d[2][13]*(grid->c3a[2][13]*mhd->sz[i][1][k]+grid->c3b[2][13]*mhd->sz[i][3][k]);
				mhd->six[i][0][k]=grid->c1[2][11]*mhd->six[i][2][k]+grid->c2[2][11]*mhd->six[i][4][k]+grid->d[2][11]*(grid->c3a[2][11]*mhd->six[i][1][k]+grid->c3b[2][11]*mhd->six[i][3][k]);
				mhd->siy[i][0][k]=grid->c1[2][12]*mhd->siy[i][2][k]+grid->c2[2][12]*mhd->siy[i][4][k]+grid->d[2][12]*(grid->c3a[2][12]*mhd->siy[i][1][k]+grid->c3b[2][12]*mhd->siy[i][3][k]);
				mhd->siz[i][0][k]=grid->c1[2][13]*mhd->siz[i][2][k]+grid->c2[2][13]*mhd->siz[i][4][k]+grid->d[2][13]*(grid->c3a[2][13]*mhd->siz[i][1][k]+grid->c3b[2][13]*mhd->siz[i][3][k]);
				mhd->snx[i][0][k]=grid->c1[2][11]*mhd->snx[i][2][k]+grid->c2[2][11]*mhd->snx[i][4][k]+grid->d[2][11]*(grid->c3a[2][11]*mhd->snx[i][1][k]+grid->c3b[2][11]*mhd->snx[i][3][k]);
				mhd->sny[i][0][k]=grid->c1[2][12]*mhd->sny[i][2][k]+grid->c2[2][12]*mhd->sny[i][4][k]+grid->d[2][12]*(grid->c3a[2][12]*mhd->sny[i][1][k]+grid->c3b[2][12]*mhd->sny[i][3][k]);
				mhd->snz[i][0][k]=grid->c1[2][13]*mhd->snz[i][2][k]+grid->c2[2][13]*mhd->snz[i][4][k]+grid->d[2][13]*(grid->c3a[2][13]*mhd->snz[i][1][k]+grid->c3b[2][13]*mhd->snz[i][3][k]);
				//Check for k=-1
				if(grid->k[2][8]==-1) mhd->bx[i][1][k]=0.;
				if(grid->k[2][9]==-1) mhd->by[i][1][k]=0.;
				if(grid->k[2][10]==-1) mhd->bz[i][1][k]=0.;
				if(grid->k[2][11]==-1) mhd->sx[i][1][k]=0.;
				if(grid->k[2][12]==-1) mhd->sy[i][1][k]=0.;
				if(grid->k[2][13]==-1) mhd->sz[i][1][k]=0.;
				if(grid->k[2][11]==-1) mhd->six[i][1][k]=0.;
				if(grid->k[2][12]==-1) mhd->siy[i][1][k]=0.;
				if(grid->k[2][13]==-1) mhd->siz[i][1][k]=0.;
				if(grid->k[2][11]==-1) mhd->snx[i][1][k]=0.;
				if(grid->k[2][12]==-1) mhd->sny[i][1][k]=0.;
				if(grid->k[2][13]==-1) mhd->snz[i][1][k]=0.;
				//Y-Max
				mhd->rho[i][grid->nym1][k]=grid->c1[3][0]*mhd->rho[i][grid->nym3][k]+grid->c2[3][0]*mhd->rho[i][grid->ny-5][k]+grid->d[3][0]*(grid->c3a[3][0]*mhd->rho[i][grid->nym2][k]+grid->c3b[3][0]*mhd->rho[i][grid->nym4][k]);
				mhd->rhoi[i][grid->nym1][k]=grid->c1[3][1]*mhd->rhoi[i][grid->nym3][k]+grid->c2[3][1]*mhd->rhoi[i][grid->ny-5][k]+grid->d[3][1]*(grid->c3a[3][1]*mhd->rhoi[i][grid->nym2][k]+grid->c3b[3][1]*mhd->rhoi[i][grid->nym4][k]);
				mhd->rhon[i][grid->nym1][k]=grid->c1[3][3]*mhd->rhon[i][grid->nym3][k]+grid->c2[3][3]*mhd->rhon[i][grid->ny-5][k]+grid->d[3][3]*(grid->c3a[3][3]*mhd->rhon[i][grid->nym2][k]+grid->c3b[3][3]*mhd->rhon[i][grid->nym4][k]);
				mhd->rhoe[i][grid->nym1][k]=mhd->me*(mhd->zi[i][grid->nym1][k]*mhd->rhoi[i][grid->nym1][k]*miinv-mhd->zd[i][grid->nym1][k]*mhd->rho[i][grid->nym1][k]*mdinv);
				mhd->p[i][grid->nym1][k]=grid->c1[3][4]*mhd->p[i][grid->nym3][k]+grid->c2[3][4]*mhd->p[i][grid->ny-5][k]+grid->d[3][4]*(grid->c3a[3][4]*mhd->p[i][grid->nym2][k]+grid->c3b[3][4]*mhd->p[i][grid->nym4][k]);
				mhd->pi[i][grid->nym1][k]=grid->c1[3][5]*mhd->pi[i][grid->nym3][k]+grid->c2[3][5]*mhd->pi[i][grid->ny-5][k]+grid->d[3][5]*(grid->c3a[3][5]*mhd->pi[i][grid->nym2][k]+grid->c3b[3][5]*mhd->pi[i][grid->nym4][k]);
				mhd->pe[i][grid->nym1][k]=grid->c1[3][6]*mhd->pe[i][grid->nym3][k]+grid->c2[3][6]*mhd->pe[i][grid->ny-5][k]+grid->d[3][6]*(grid->c3a[3][6]*mhd->pe[i][grid->nym2][k]+grid->c3b[3][6]*mhd->pe[i][grid->nym4][k]);
				mhd->pn[i][grid->nym1][k]=grid->c1[3][7]*mhd->pn[i][grid->nym3][k]+grid->c2[3][7]*mhd->pn[i][grid->ny-5][k]+grid->d[3][7]*(grid->c3a[3][7]*mhd->pn[i][grid->nym2][k]+grid->c3b[3][7]*mhd->pn[i][grid->nym4][k]);
				mhd->bx[i][grid->nym1][k]=grid->c1[3][8]*mhd->bx[i][grid->nym3][k]+grid->c2[3][8]*mhd->bx[i][grid->ny-5][k]+grid->d[3][8]*(grid->c3a[3][8]*mhd->bx[i][grid->nym2][k]+grid->c3b[3][8]*mhd->bx[i][grid->nym4][k]);
				mhd->by[i][grid->nym1][k]=grid->c1[3][9]*mhd->by[i][grid->nym3][k]+grid->c2[3][9]*mhd->by[i][grid->ny-5][k]+grid->d[3][9]*(grid->c3a[3][9]*mhd->by[i][grid->nym2][k]+grid->c3b[3][9]*mhd->by[i][grid->nym4][k]);
				mhd->bz[i][grid->nym1][k]=grid->c1[3][10]*mhd->bz[i][grid->nym3][k]+grid->c2[3][10]*mhd->bz[i][grid->ny-5][k]+grid->d[3][10]*(grid->c3a[3][10]*mhd->bz[i][grid->nym2][k]+grid->c3b[3][10]*mhd->bz[i][grid->nym4][k]);
				mhd->sx[i][grid->nym1][k]=grid->c1[3][11]*mhd->sx[i][grid->nym3][k]+grid->c2[3][11]*mhd->sx[i][grid->ny-5][k]+grid->d[3][11]*(grid->c3a[3][11]*mhd->sx[i][grid->nym2][k]+grid->c3b[3][11]*mhd->sx[i][grid->nym4][k]);
				mhd->sy[i][grid->nym1][k]=grid->c1[3][12]*mhd->sy[i][grid->nym3][k]+grid->c2[3][12]*mhd->sy[i][grid->ny-5][k]+grid->d[3][12]*(grid->c3a[3][12]*mhd->sy[i][grid->nym2][k]+grid->c3b[3][12]*mhd->sy[i][grid->nym4][k]);
				mhd->sz[i][grid->nym1][k]=grid->c1[3][13]*mhd->sz[i][grid->nym3][k]+grid->c2[3][13]*mhd->sz[i][grid->ny-5][k]+grid->d[3][13]*(grid->c3a[3][13]*mhd->sz[i][grid->nym2][k]+grid->c3b[3][13]*mhd->sz[i][grid->nym4][k]);
				mhd->six[i][grid->nym1][k]=grid->c1[3][11]*mhd->six[i][grid->nym3][k]+grid->c2[3][11]*mhd->six[i][grid->ny-5][k]+grid->d[3][11]*(grid->c3a[3][11]*mhd->six[i][grid->nym2][k]+grid->c3b[3][11]*mhd->six[i][grid->nym4][k]);
				mhd->siy[i][grid->nym1][k]=grid->c1[3][12]*mhd->siy[i][grid->nym3][k]+grid->c2[3][12]*mhd->siy[i][grid->ny-5][k]+grid->d[3][12]*(grid->c3a[3][12]*mhd->siy[i][grid->nym2][k]+grid->c3b[3][12]*mhd->siy[i][grid->nym4][k]);
				mhd->siz[i][grid->nym1][k]=grid->c1[3][13]*mhd->siz[i][grid->nym3][k]+grid->c2[3][13]*mhd->siz[i][grid->ny-5][k]+grid->d[3][13]*(grid->c3a[3][13]*mhd->siz[i][grid->nym2][k]+grid->c3b[3][13]*mhd->siz[i][grid->nym4][k]);
				mhd->snx[i][grid->nym1][k]=grid->c1[3][11]*mhd->snx[i][grid->nym3][k]+grid->c2[3][11]*mhd->snx[i][grid->ny-5][k]+grid->d[3][11]*(grid->c3a[3][11]*mhd->snx[i][grid->nym2][k]+grid->c3b[3][11]*mhd->snx[i][grid->nym4][k]);
				mhd->sny[i][grid->nym1][k]=grid->c1[3][12]*mhd->sny[i][grid->nym3][k]+grid->c2[3][12]*mhd->sny[i][grid->ny-5][k]+grid->d[3][12]*(grid->c3a[3][12]*mhd->sny[i][grid->nym2][k]+grid->c3b[3][12]*mhd->sny[i][grid->nym4][k]);
				mhd->snz[i][grid->nym1][k]=grid->c1[3][13]*mhd->snz[i][grid->nym3][k]+grid->c2[3][13]*mhd->snz[i][grid->ny-5][k]+grid->d[3][13]*(grid->c3a[3][13]*mhd->snz[i][grid->nym2][k]+grid->c3b[3][13]*mhd->snz[i][grid->nym4][k]);
				//Check for k=-1
				if(grid->k[3][8]==-1) mhd->bx[i][grid->nym2][k]=0.;
				if(grid->k[3][9]==-1) mhd->by[i][grid->nym2][k]=0.;
				if(grid->k[3][10]==-1) mhd->bz[i][grid->nym2][k]=0.;
				if(grid->k[3][11]==-1) mhd->sx[i][grid->nym2][k]=0.;
				if(grid->k[3][12]==-1) mhd->sy[i][grid->nym2][k]=0.;
				if(grid->k[3][13]==-1) mhd->sz[i][grid->nym2][k]=0.;
				if(grid->k[3][11]==-1) mhd->six[i][grid->nym2][k]=0.;
				if(grid->k[3][12]==-1) mhd->siy[i][grid->nym2][k]=0.;
				if(grid->k[3][13]==-1) mhd->siz[i][grid->nym2][k]=0.;
				if(grid->k[3][11]==-1) mhd->snx[i][grid->nym2][k]=0.;
				if(grid->k[3][12]==-1) mhd->sny[i][grid->nym2][k]=0.;
				if(grid->k[3][13]==-1) mhd->snz[i][grid->nym2][k]=0.;
			}
		}
		#pragma omp barrier
	}
//	Y Time Dependant BC's
	else if (grid->pery == 2)
	{
		#pragma omp barrier
	}
//      Z-BC
	if (grid->perz == 1)
	{
//		Z-min
		#pragma omp for nowait
		for(i=1;i<grid->nxm1;i++)
		{
			for(j=1;j<grid->nym1;j++)
			{
				mhd-> rho[i][j][0]=mhd-> rho[i][j][grid->nzm3];
				mhd->rhoi[i][j][0]=mhd->rhoi[i][j][grid->nzm3];
				mhd->rhoe[i][j][0]=mhd->rhoe[i][j][grid->nzm3];
				mhd->rhon[i][j][0]=mhd->rhon[i][j][grid->nzm3];
				mhd->  sx[i][j][0]=mhd->  sx[i][j][grid->nzm3];
				mhd->  sy[i][j][0]=mhd->  sy[i][j][grid->nzm3];
				mhd->  sz[i][j][0]=mhd->  sz[i][j][grid->nzm3];
				mhd-> six[i][j][0]=mhd-> six[i][j][grid->nzm3];
				mhd-> siy[i][j][0]=mhd-> siy[i][j][grid->nzm3];
				mhd-> siz[i][j][0]=mhd-> siz[i][j][grid->nzm3];
				mhd-> snx[i][j][0]=mhd-> snx[i][j][grid->nzm3];
				mhd-> sny[i][j][0]=mhd-> sny[i][j][grid->nzm3];
				mhd-> snz[i][j][0]=mhd-> snz[i][j][grid->nzm3];
				mhd->  bx[i][j][0]=mhd->  bx[i][j][grid->nzm3];
				mhd->  by[i][j][0]=mhd->  by[i][j][grid->nzm3];
				mhd->  bz[i][j][0]=mhd->  bz[i][j][grid->nzm3];
				mhd->   p[i][j][0]=mhd->   p[i][j][grid->nzm3];
				mhd->  pi[i][j][0]=mhd->  pi[i][j][grid->nzm3];
				mhd->  pe[i][j][0]=mhd->  pe[i][j][grid->nzm3];
				mhd->  pn[i][j][0]=mhd->  pn[i][j][grid->nzm3];
			}
		}
//		Z-max
		#pragma omp for
		for(i=1;i<grid->nxm1;i++)
		{
			for(j=1;j<grid->nym1;j++)
			{
				mhd-> rho[i][j][grid->nzm1]=mhd-> rho[i][j][2];
				mhd->rhoi[i][j][grid->nzm1]=mhd->rhoi[i][j][2];
				mhd->rhoe[i][j][grid->nzm1]=mhd->rhoe[i][j][2];
				mhd->rhon[i][j][grid->nzm1]=mhd->rhon[i][j][2];
				mhd->  sx[i][j][grid->nzm1]=mhd->  sx[i][j][2];
				mhd->  sy[i][j][grid->nzm1]=mhd->  sy[i][j][2];
				mhd->  sz[i][j][grid->nzm1]=mhd->  sz[i][j][2];
				mhd-> six[i][j][grid->nzm1]=mhd-> six[i][j][2];
				mhd-> siy[i][j][grid->nzm1]=mhd-> siy[i][j][2];
				mhd-> siz[i][j][grid->nzm1]=mhd-> siz[i][j][2];
				mhd-> snx[i][j][grid->nzm1]=mhd-> snx[i][j][2];
				mhd-> sny[i][j][grid->nzm1]=mhd-> sny[i][j][2];
				mhd-> snz[i][j][grid->nzm1]=mhd-> snz[i][j][2];
				mhd->  bx[i][j][grid->nzm1]=mhd->  bx[i][j][2];
				mhd->  by[i][j][grid->nzm1]=mhd->  by[i][j][2];
				mhd->  bz[i][j][grid->nzm1]=mhd->  bz[i][j][2];
				mhd->   p[i][j][grid->nzm1]=mhd->   p[i][j][2];
				mhd->  pi[i][j][grid->nzm1]=mhd->  pi[i][j][2];
				mhd->  pe[i][j][grid->nzm1]=mhd->  pe[i][j][2];
				mhd->  pn[i][j][grid->nzm1]=mhd->  pn[i][j][2];
			}
		}
		#pragma omp barrier
	}
	else if (grid->perz == 0)
	{
		#pragma omp for nowait
		for(i=1;i<grid->nxm1;i++)
		{
			for(j=1;j<grid->nym1;j++)
			{
				//Z-Min				
				mhd->rho[i][j][0]=grid->c1[4][0]*mhd->rho[i][j][2]+grid->c2[4][0]*mhd->rho[i][j][4]+grid->d[4][0]*(grid->c3a[4][0]*mhd->rho[i][j][1]+grid->c3b[4][0]*mhd->rho[i][j][3]);
				mhd->rhoi[i][j][0]=grid->c1[4][1]*mhd->rhoi[i][j][2]+grid->c2[4][1]*mhd->rhoi[i][j][4]+grid->d[4][1]*(grid->c3a[4][1]*mhd->rhoi[i][j][1]+grid->c3b[4][1]*mhd->rhoi[i][j][3]);
				mhd->rhon[i][j][0]=grid->c1[4][3]*mhd->rhon[i][j][2]+grid->c2[4][3]*mhd->rhon[i][j][4]+grid->d[4][3]*(grid->c3a[4][3]*mhd->rhon[i][j][1]+grid->c3b[4][3]*mhd->rhon[i][j][3]);
				mhd->rhoe[i][j][0]=mhd->me*(mhd->zi[i][j][0]*mhd->rhoi[i][j][0]*miinv-mhd->zd[i][j][0]*mhd->rho[i][j][0]*mdinv);
				mhd->p[i][j][0]=grid->c1[4][4]*mhd->p[i][j][2]+grid->c2[4][4]*mhd->p[i][j][4]+grid->d[4][4]*(grid->c3a[4][4]*mhd->p[i][j][1]+grid->c3b[4][4]*mhd->p[i][j][3]);
				mhd->pi[i][j][0]=grid->c1[4][5]*mhd->pi[i][j][2]+grid->c2[4][5]*mhd->pi[i][j][4]+grid->d[4][5]*(grid->c3a[4][5]*mhd->pi[i][j][1]+grid->c3b[4][5]*mhd->pi[i][j][3]);
				mhd->pe[i][j][0]=grid->c1[4][6]*mhd->pe[i][j][2]+grid->c2[4][6]*mhd->pe[i][j][4]+grid->d[4][6]*(grid->c3a[4][6]*mhd->pe[i][j][1]+grid->c3b[4][6]*mhd->pe[i][j][3]);
				mhd->pn[i][j][0]=grid->c1[4][7]*mhd->pn[i][j][2]+grid->c2[4][7]*mhd->pn[i][j][4]+grid->d[4][7]*(grid->c3a[4][7]*mhd->pn[i][j][1]+grid->c3b[4][7]*mhd->pn[i][j][3]);
				mhd->bx[i][j][0]=grid->c1[4][8]*mhd->bx[i][j][2]+grid->c2[4][8]*mhd->bx[i][j][4]+grid->d[4][8]*(grid->c3a[4][8]*mhd->bx[i][j][1]+grid->c3b[4][8]*mhd->bx[i][j][3]);
				mhd->by[i][j][0]=grid->c1[4][9]*mhd->by[i][j][2]+grid->c2[4][9]*mhd->by[i][j][4]+grid->d[4][9]*(grid->c3a[4][9]*mhd->by[i][j][1]+grid->c3b[4][9]*mhd->by[i][j][3]);
				mhd->bz[i][j][0]=grid->c1[4][10]*mhd->bz[i][j][2]+grid->c2[4][10]*mhd->bz[i][j][4]+grid->d[4][10]*(grid->c3a[4][10]*mhd->bz[i][j][1]+grid->c3b[4][10]*mhd->bz[i][j][3]);
				mhd->sx[i][j][0]=grid->c1[4][11]*mhd->sx[i][j][2]+grid->c2[4][11]*mhd->sx[i][j][4]+grid->d[4][11]*(grid->c3a[4][11]*mhd->sx[i][j][1]+grid->c3b[4][11]*mhd->sx[i][j][3]);
				mhd->sy[i][j][0]=grid->c1[4][12]*mhd->sy[i][j][2]+grid->c2[4][12]*mhd->sy[i][j][4]+grid->d[4][12]*(grid->c3a[4][12]*mhd->sy[i][j][1]+grid->c3b[4][12]*mhd->sy[i][j][3]);
				mhd->sz[i][j][0]=grid->c1[4][13]*mhd->sz[i][j][2]+grid->c2[4][13]*mhd->sz[i][j][4]+grid->d[4][13]*(grid->c3a[4][13]*mhd->sz[i][j][1]+grid->c3b[4][13]*mhd->sz[i][j][3]);
				mhd->six[i][j][0]=grid->c1[4][11]*mhd->six[i][j][2]+grid->c2[4][11]*mhd->six[i][j][4]+grid->d[4][11]*(grid->c3a[4][11]*mhd->six[i][j][1]+grid->c3b[4][11]*mhd->six[i][j][3]);
				mhd->siy[i][j][0]=grid->c1[4][12]*mhd->siy[i][j][2]+grid->c2[4][12]*mhd->siy[i][j][4]+grid->d[4][12]*(grid->c3a[4][12]*mhd->siy[i][j][1]+grid->c3b[4][12]*mhd->siy[i][j][3]);
				mhd->siz[i][j][0]=grid->c1[4][13]*mhd->siz[i][j][2]+grid->c2[4][13]*mhd->siz[i][j][4]+grid->d[4][13]*(grid->c3a[4][13]*mhd->siz[i][j][1]+grid->c3b[4][13]*mhd->siz[i][j][3]);
				mhd->snx[i][j][0]=grid->c1[4][11]*mhd->snx[i][j][2]+grid->c2[4][11]*mhd->snx[i][j][4]+grid->d[4][11]*(grid->c3a[4][11]*mhd->snx[i][j][1]+grid->c3b[4][11]*mhd->snx[i][j][3]);
				mhd->sny[i][j][0]=grid->c1[4][12]*mhd->sny[i][j][2]+grid->c2[4][12]*mhd->sny[i][j][4]+grid->d[4][12]*(grid->c3a[4][12]*mhd->sny[i][j][1]+grid->c3b[4][12]*mhd->sny[i][j][3]);
				mhd->snz[i][j][0]=grid->c1[4][13]*mhd->snz[i][j][2]+grid->c2[4][13]*mhd->snz[i][j][4]+grid->d[4][13]*(grid->c3a[4][13]*mhd->snz[i][j][1]+grid->c3b[4][13]*mhd->snz[i][j][3]);
				//Check for k=-1
				if(grid->k[4][8]==-1) mhd->bx[i][j][1]=0.;
				if(grid->k[4][9]==-1) mhd->by[i][j][1]=0.;
				if(grid->k[4][10]==-1) mhd->bz[i][j][1]=0.;
				if(grid->k[4][11]==-1) mhd->sx[i][j][1]=0.;
				if(grid->k[4][12]==-1) mhd->sy[i][j][1]=0.;
				if(grid->k[4][13]==-1) mhd->sz[i][j][1]=0.;
				if(grid->k[4][11]==-1) mhd->six[i][j][1]=0.;
				if(grid->k[4][12]==-1) mhd->siy[i][j][1]=0.;
				if(grid->k[4][13]==-1) mhd->siz[i][j][1]=0.;
				if(grid->k[4][11]==-1) mhd->snx[i][j][1]=0.;
				if(grid->k[4][12]==-1) mhd->sny[i][j][1]=0.;
				if(grid->k[4][13]==-1) mhd->snz[i][j][1]=0.;
				//Z-Max
				mhd->rho[i][j][grid->nzm1]=grid->c1[5][0]*mhd->rho[i][j][grid->nzm3]+grid->c2[5][0]*mhd->rho[i][j][grid->nz-5]+grid->d[5][0]*(grid->c3a[5][0]*mhd->rho[i][j][grid->nzm2]+grid->c3b[5][0]*mhd->rho[i][j][grid->nzm4]);
				mhd->rhoi[i][j][grid->nzm1]=grid->c1[5][1]*mhd->rhoi[i][j][grid->nzm3]+grid->c2[5][1]*mhd->rhoi[i][j][grid->nz-5]+grid->d[5][1]*(grid->c3a[5][1]*mhd->rhoi[i][j][grid->nzm2]+grid->c3b[5][1]*mhd->rhoi[i][j][grid->nzm4]);
				mhd->rhon[i][j][grid->nzm1]=grid->c1[5][3]*mhd->rhon[i][j][grid->nzm3]+grid->c2[5][3]*mhd->rhon[i][j][grid->nz-5]+grid->d[5][3]*(grid->c3a[5][3]*mhd->rhon[i][j][grid->nzm2]+grid->c3b[5][3]*mhd->rhon[i][j][grid->nzm4]);
				mhd->rhoe[i][j][grid->nzm1]=mhd->me*(mhd->zi[i][j][grid->nzm1]*mhd->rhoi[i][j][grid->nzm1]*miinv-mhd->zd[i][j][grid->nzm1]*mhd->rho[i][j][grid->nzm1]*mdinv);
				mhd->p[i][j][grid->nzm1]=grid->c1[5][4]*mhd->p[i][j][grid->nzm3]+grid->c2[5][4]*mhd->p[i][j][grid->nz-5]+grid->d[5][4]*(grid->c3a[5][4]*mhd->p[i][j][grid->nzm2]+grid->c3b[5][4]*mhd->p[i][j][grid->nzm4]);
				mhd->pi[i][j][grid->nzm1]=grid->c1[5][5]*mhd->pi[i][j][grid->nzm3]+grid->c2[5][5]*mhd->pi[i][j][grid->nz-5]+grid->d[5][5]*(grid->c3a[5][5]*mhd->pi[i][j][grid->nzm2]+grid->c3b[5][5]*mhd->pi[i][j][grid->nzm4]);
				mhd->pe[i][j][grid->nzm1]=grid->c1[5][6]*mhd->pe[i][j][grid->nzm3]+grid->c2[5][6]*mhd->pe[i][j][grid->nz-5]+grid->d[5][6]*(grid->c3a[5][6]*mhd->pe[i][j][grid->nzm2]+grid->c3b[5][6]*mhd->pe[i][j][grid->nzm4]);
				mhd->pn[i][j][grid->nzm1]=grid->c1[5][7]*mhd->pn[i][j][grid->nzm3]+grid->c2[5][7]*mhd->pn[i][j][grid->nz-5]+grid->d[5][7]*(grid->c3a[5][7]*mhd->pn[i][j][grid->nzm2]+grid->c3b[5][7]*mhd->pn[i][j][grid->nzm4]);
				mhd->bx[i][j][grid->nzm1]=grid->c1[5][8]*mhd->bx[i][j][grid->nzm3]+grid->c2[5][8]*mhd->bx[i][j][grid->nz-5]+grid->d[5][8]*(grid->c3a[5][8]*mhd->bx[i][j][grid->nzm2]+grid->c3b[5][8]*mhd->bx[i][j][grid->nzm4]);
				mhd->by[i][j][grid->nzm1]=grid->c1[5][9]*mhd->by[i][j][grid->nzm3]+grid->c2[5][9]*mhd->by[i][j][grid->nz-5]+grid->d[5][9]*(grid->c3a[5][9]*mhd->by[i][j][grid->nzm2]+grid->c3b[5][9]*mhd->by[i][j][grid->nzm4]);
				mhd->bz[i][j][grid->nzm1]=grid->c1[5][10]*mhd->bz[i][j][grid->nzm3]+grid->c2[5][10]*mhd->bz[i][j][grid->nz-5]+grid->d[5][10]*(grid->c3a[5][10]*mhd->bz[i][j][grid->nzm2]+grid->c3b[5][10]*mhd->bz[i][j][grid->nzm4]);
				mhd->sx[i][j][grid->nzm1]=grid->c1[5][11]*mhd->sx[i][j][grid->nzm3]+grid->c2[5][11]*mhd->sx[i][j][grid->nz-5]+grid->d[5][11]*(grid->c3a[5][11]*mhd->sx[i][j][grid->nzm2]+grid->c3b[5][11]*mhd->sx[i][j][grid->nzm4]);
				mhd->sy[i][j][grid->nzm1]=grid->c1[5][12]*mhd->sy[i][j][grid->nzm3]+grid->c2[5][12]*mhd->sy[i][j][grid->nz-5]+grid->d[5][12]*(grid->c3a[5][12]*mhd->sy[i][j][grid->nzm2]+grid->c3b[5][12]*mhd->sy[i][j][grid->nzm4]);
				mhd->sz[i][j][grid->nzm1]=grid->c1[5][13]*mhd->sz[i][j][grid->nzm3]+grid->c2[5][13]*mhd->sz[i][j][grid->nz-5]+grid->d[5][13]*(grid->c3a[5][13]*mhd->sz[i][j][grid->nzm2]+grid->c3b[5][13]*mhd->sz[i][j][grid->nzm4]);
				mhd->six[i][j][grid->nzm1]=grid->c1[5][11]*mhd->six[i][j][grid->nzm3]+grid->c2[5][11]*mhd->six[i][j][grid->nz-5]+grid->d[5][11]*(grid->c3a[5][11]*mhd->six[i][j][grid->nzm2]+grid->c3b[5][11]*mhd->six[i][j][grid->nzm4]);
				mhd->siy[i][j][grid->nzm1]=grid->c1[5][12]*mhd->siy[i][j][grid->nzm3]+grid->c2[5][12]*mhd->siy[i][j][grid->nz-5]+grid->d[5][12]*(grid->c3a[5][12]*mhd->siy[i][j][grid->nzm2]+grid->c3b[5][12]*mhd->siy[i][j][grid->nzm4]);
				mhd->siz[i][j][grid->nzm1]=grid->c1[5][13]*mhd->siz[i][j][grid->nzm3]+grid->c2[5][13]*mhd->siz[i][j][grid->nz-5]+grid->d[5][13]*(grid->c3a[5][13]*mhd->siz[i][j][grid->nzm2]+grid->c3b[5][13]*mhd->siz[i][j][grid->nzm4]);
				mhd->snx[i][j][grid->nzm1]=grid->c1[5][11]*mhd->snx[i][j][grid->nzm3]+grid->c2[5][11]*mhd->snx[i][j][grid->nz-5]+grid->d[5][11]*(grid->c3a[5][11]*mhd->snx[i][j][grid->nzm2]+grid->c3b[5][11]*mhd->snx[i][j][grid->nzm4]);
				mhd->sny[i][j][grid->nzm1]=grid->c1[5][12]*mhd->sny[i][j][grid->nzm3]+grid->c2[5][12]*mhd->sny[i][j][grid->nz-5]+grid->d[5][12]*(grid->c3a[5][12]*mhd->sny[i][j][grid->nzm2]+grid->c3b[5][12]*mhd->sny[i][j][grid->nzm4]);
				mhd->snz[i][j][grid->nzm1]=grid->c1[5][13]*mhd->snz[i][j][grid->nzm3]+grid->c2[5][13]*mhd->snz[i][j][grid->nz-5]+grid->d[5][13]*(grid->c3a[5][13]*mhd->snz[i][j][grid->nzm2]+grid->c3b[5][13]*mhd->snz[i][j][grid->nzm4]);
				//Check for k=-1
				if(grid->k[5][8]==-1) mhd->bx[i][j][grid->nzm2]=0.;
				if(grid->k[5][9]==-1) mhd->by[i][j][grid->nzm2]=0.;
				if(grid->k[5][10]==-1) mhd->bz[i][j][grid->nzm2]=0.;
				if(grid->k[5][11]==-1) mhd->sx[i][j][grid->nzm2]=0.;
				if(grid->k[5][12]==-1) mhd->sy[i][j][grid->nzm2]=0.;
				if(grid->k[5][13]==-1) mhd->sz[i][j][grid->nzm2]=0.;
				if(grid->k[5][11]==-1) mhd->six[i][j][grid->nzm2]=0.;
				if(grid->k[5][12]==-1) mhd->siy[i][j][grid->nzm2]=0.;
				if(grid->k[5][13]==-1) mhd->siz[i][j][grid->nzm2]=0.;
				if(grid->k[5][11]==-1) mhd->snx[i][j][grid->nzm2]=0.;
				if(grid->k[5][12]==-1) mhd->sny[i][j][grid->nzm2]=0.;
				if(grid->k[5][13]==-1) mhd->snz[i][j][grid->nzm2]=0.;
			}
		}
		#pragma omp barrier
	}
//	Z Time Dependant BC's
	else if (grid->perz == 2)
	{
		#pragma omp single
		{
		//double freq=M_PI;
		//double bz0=1.0;
		//double db=0.001;
		//double sin_phase=sin(freq*(*time));
		//double cos_phase=cos(freq*(*time));
		for(i=0;i<grid->nx;i++)
		{
			for(j=0;j<grid->ny;j++)
			{	
				//Z-Min
				//Magnetic Perturbation
				//mhd->bx[i][j][0]=db*sin_phase;
				//mhd->by[i][j][0]=2.*mhd->by[i][j][1]-mhd->by[i][j][2];
				//mhd->bz[i][j][0]=bz0;
				//Numerical Boundary
				mhd->rho[i][j][0]=grid->c1[4][0]*mhd->rho[i][j][2]+grid->c2[4][0]*mhd->rho[i][j][4]+grid->d[4][0]*(grid->c3a[4][0]*mhd->rho[i][j][1]+grid->c3b[4][0]*mhd->rho[i][j][3]);
				mhd->rhoi[i][j][0]=grid->c1[4][1]*mhd->rhoi[i][j][2]+grid->c2[4][1]*mhd->rhoi[i][j][4]+grid->d[4][1]*(grid->c3a[4][1]*mhd->rhoi[i][j][1]+grid->c3b[4][1]*mhd->rhoi[i][j][3]);
				mhd->rhon[i][j][0]=grid->c1[4][3]*mhd->rhon[i][j][2]+grid->c2[4][3]*mhd->rhon[i][j][4]+grid->d[4][3]*(grid->c3a[4][3]*mhd->rhon[i][j][1]+grid->c3b[4][3]*mhd->rhon[i][j][3]);
				mhd->rhoe[i][j][0]=mhd->me*(mhd->zi[i][j][0]*mhd->rhoi[i][j][0]*miinv-mhd->zd[i][j][0]*mhd->rho[i][j][0]);
				mhd->p[i][j][0]=grid->c1[4][4]*mhd->p[i][j][2]+grid->c2[4][4]*mhd->p[i][j][4]+grid->d[4][4]*(grid->c3a[4][4]*mhd->p[i][j][1]+grid->c3b[4][4]*mhd->p[i][j][3]);
				mhd->pi[i][j][0]=grid->c1[4][5]*mhd->pi[i][j][2]+grid->c2[4][5]*mhd->pi[i][j][4]+grid->d[4][5]*(grid->c3a[4][5]*mhd->pi[i][j][1]+grid->c3b[4][5]*mhd->pi[i][j][3]);
				mhd->pe[i][j][0]=grid->c1[4][6]*mhd->pe[i][j][2]+grid->c2[4][6]*mhd->pe[i][j][4]+grid->d[4][6]*(grid->c3a[4][6]*mhd->pe[i][j][1]+grid->c3b[4][6]*mhd->pe[i][j][3]);
				mhd->pn[i][j][0]=grid->c1[4][7]*mhd->pn[i][j][2]+grid->c2[4][7]*mhd->pn[i][j][4]+grid->d[4][7]*(grid->c3a[4][7]*mhd->pn[i][j][1]+grid->c3b[4][7]*mhd->pn[i][j][3]);
				mhd->sx[i][j][0]=2.*mhd->sx[i][j][1]-mhd->sx[i][j][2];
				mhd->sy[i][j][0]=2.*mhd->sy[i][j][1]-mhd->sy[i][j][2];
				mhd->sz[i][j][0]=2.*mhd->sz[i][j][1]-mhd->sz[i][j][2];
				//mhd->sz[i][j][0]=grid->c1[4][13]*mhd->sz[i][j][2]+grid->c2[4][13]*mhd->sz[i][j][4]+grid->d[4][13]*(grid->c3a[4][13]*mhd->sz[i][j][1]+grid->c3b[4][13]*mhd->sz[i][j][3]);
				mhd->six[i][j][0]=2.*mhd->six[i][j][1]-mhd->six[i][j][2];
				mhd->siy[i][j][0]=2.*mhd->siy[i][j][1]-mhd->siy[i][j][2];
				mhd->siz[i][j][0]=2.*mhd->siz[i][j][1]-mhd->siz[i][j][2];
				//mhd->siz[i][j][0]=grid->c1[4][16]*mhd->siz[i][j][2]+grid->c2[4][16]*mhd->siz[i][j][4]+grid->d[4][16]*(grid->c3a[4][16]*mhd->siz[i][j][1]+grid->c3b[4][16]*mhd->siz[i][j][3]);
				mhd->snx[i][j][0]=grid->c1[4][17]*mhd->snx[i][j][2]+grid->c2[4][17]*mhd->snx[i][j][4]+grid->d[4][17]*(grid->c3a[4][17]*mhd->snx[i][j][1]+grid->c3b[4][17]*mhd->snx[i][j][3]);
				mhd->sny[i][j][0]=grid->c1[4][18]*mhd->sny[i][j][2]+grid->c2[4][18]*mhd->sny[i][j][4]+grid->d[4][18]*(grid->c3a[4][18]*mhd->sny[i][j][1]+grid->c3b[4][18]*mhd->sny[i][j][3]);
				mhd->snz[i][j][0]=grid->c1[4][19]*mhd->snz[i][j][2]+grid->c2[4][19]*mhd->snz[i][j][4]+grid->d[4][19]*(grid->c3a[4][19]*mhd->snz[i][j][1]+grid->c3b[4][19]*mhd->snz[i][j][3]);
				mhd->bx[i][j][0]=2.*mhd->bx[i][j][1]-mhd->bx[i][j][2];
				mhd->by[i][j][0]=2.*mhd->by[i][j][1]-mhd->by[i][j][2];
				mhd->bz[i][j][0]=2.*mhd->bz[i][j][1]-mhd->bz[i][j][2];

				//Z-Max
				mhd->rho[i][j][grid->nzm1]=grid->c1[5][0]*mhd->rho[i][j][grid->nzm3]+grid->c2[5][0]*mhd->rho[i][j][grid->nz-5]+grid->d[5][0]*(grid->c3a[5][0]*mhd->rho[i][j][grid->nzm2]+grid->c3b[5][0]*mhd->rho[i][j][grid->nzm4]);
				mhd->rhoi[i][j][grid->nzm1]=grid->c1[5][1]*mhd->rhoi[i][j][grid->nzm3]+grid->c2[5][1]*mhd->rhoi[i][j][grid->nz-5]+grid->d[5][1]*(grid->c3a[5][1]*mhd->rhoi[i][j][grid->nzm2]+grid->c3b[5][1]*mhd->rhoi[i][j][grid->nzm4]);
				mhd->rhon[i][j][grid->nzm1]=grid->c1[5][3]*mhd->rhon[i][j][grid->nzm3]+grid->c2[5][3]*mhd->rhon[i][j][grid->nz-5]+grid->d[5][3]*(grid->c3a[5][3]*mhd->rhon[i][j][grid->nzm2]+grid->c3b[5][3]*mhd->rhon[i][j][grid->nzm4]);
				mhd->rhoe[i][j][grid->nzm1]=mhd->me*(mhd->zi[i][j][grid->nzm1]*mhd->rhoi[i][j][grid->nzm1]*miinv-mhd->zd[i][j][grid->nzm1]*mhd->rho[i][j][grid->nzm1]);
				mhd->p[i][j][grid->nzm1]=grid->c1[5][4]*mhd->p[i][j][grid->nzm3]+grid->c2[5][4]*mhd->p[i][j][grid->nz-5]+grid->d[5][4]*(grid->c3a[5][4]*mhd->p[i][j][grid->nzm2]+grid->c3b[5][4]*mhd->p[i][j][grid->nzm4]);
				mhd->pi[i][j][grid->nzm1]=grid->c1[5][5]*mhd->pi[i][j][grid->nzm3]+grid->c2[5][5]*mhd->pi[i][j][grid->nz-5]+grid->d[5][5]*(grid->c3a[5][5]*mhd->pi[i][j][grid->nzm2]+grid->c3b[5][5]*mhd->pi[i][j][grid->nzm4]);
				mhd->pe[i][j][grid->nzm1]=grid->c1[5][6]*mhd->pe[i][j][grid->nzm3]+grid->c2[5][6]*mhd->pe[i][j][grid->nz-5]+grid->d[5][6]*(grid->c3a[5][6]*mhd->pe[i][j][grid->nzm2]+grid->c3b[5][6]*mhd->pe[i][j][grid->nzm4]);
				mhd->pn[i][j][grid->nzm1]=grid->c1[5][7]*mhd->pn[i][j][grid->nzm3]+grid->c2[5][7]*mhd->pn[i][j][grid->nz-5]+grid->d[5][7]*(grid->c3a[5][7]*mhd->pn[i][j][grid->nzm2]+grid->c3b[5][7]*mhd->pn[i][j][grid->nzm4]);
				mhd->bx[i][j][grid->nzm1]=grid->c1[5][8]*mhd->bx[i][j][grid->nzm3]+grid->c2[5][8]*mhd->bx[i][j][grid->nz-5]+grid->d[5][8]*(grid->c3a[5][8]*mhd->bx[i][j][grid->nzm2]+grid->c3b[5][8]*mhd->bx[i][j][grid->nzm4]);
				mhd->by[i][j][grid->nzm1]=grid->c1[5][9]*mhd->by[i][j][grid->nzm3]+grid->c2[5][9]*mhd->by[i][j][grid->nz-5]+grid->d[5][9]*(grid->c3a[5][9]*mhd->by[i][j][grid->nzm2]+grid->c3b[5][9]*mhd->by[i][j][grid->nzm4]);
				mhd->bz[i][j][grid->nzm1]=grid->c1[5][10]*mhd->bz[i][j][grid->nzm3]+grid->c2[5][10]*mhd->bz[i][j][grid->nz-5]+grid->d[5][10]*(grid->c3a[5][10]*mhd->bz[i][j][grid->nzm2]+grid->c3b[5][10]*mhd->bz[i][j][grid->nzm4]);
				mhd->sx[i][j][grid->nzm1]=grid->c1[5][11]*mhd->sx[i][j][grid->nzm3]+grid->c2[5][11]*mhd->sx[i][j][grid->nz-5]+grid->d[5][11]*(grid->c3a[5][11]*mhd->sx[i][j][grid->nzm2]+grid->c3b[5][11]*mhd->sx[i][j][grid->nzm4]);
				mhd->sy[i][j][grid->nzm1]=grid->c1[5][12]*mhd->sy[i][j][grid->nzm3]+grid->c2[5][12]*mhd->sy[i][j][grid->nz-5]+grid->d[5][12]*(grid->c3a[5][12]*mhd->sy[i][j][grid->nzm2]+grid->c3b[5][12]*mhd->sy[i][j][grid->nzm4]);
				mhd->sz[i][j][grid->nzm1]=grid->c1[5][13]*mhd->sz[i][j][grid->nzm3]+grid->c2[5][13]*mhd->sz[i][j][grid->nz-5]+grid->d[5][13]*(grid->c3a[5][13]*mhd->sz[i][j][grid->nzm2]+grid->c3b[5][13]*mhd->sz[i][j][grid->nzm4]);
				mhd->six[i][j][grid->nzm1]=grid->c1[5][14]*mhd->six[i][j][grid->nzm3]+grid->c2[5][14]*mhd->six[i][j][grid->nz-5]+grid->d[5][14]*(grid->c3a[5][14]*mhd->six[i][j][grid->nzm2]+grid->c3b[5][14]*mhd->six[i][j][grid->nzm4]);
				mhd->siy[i][j][grid->nzm1]=grid->c1[5][15]*mhd->siy[i][j][grid->nzm3]+grid->c2[5][15]*mhd->siy[i][j][grid->nz-5]+grid->d[5][15]*(grid->c3a[5][15]*mhd->siy[i][j][grid->nzm2]+grid->c3b[5][15]*mhd->siy[i][j][grid->nzm4]);
				mhd->siz[i][j][grid->nzm1]=grid->c1[5][16]*mhd->siz[i][j][grid->nzm3]+grid->c2[5][16]*mhd->siz[i][j][grid->nz-5]+grid->d[5][16]*(grid->c3a[5][16]*mhd->siz[i][j][grid->nzm2]+grid->c3b[5][16]*mhd->siz[i][j][grid->nzm4]);
				mhd->snx[i][j][grid->nzm1]=grid->c1[5][17]*mhd->snx[i][j][grid->nzm3]+grid->c2[5][17]*mhd->snx[i][j][grid->nz-5]+grid->d[5][17]*(grid->c3a[5][17]*mhd->snx[i][j][grid->nzm2]+grid->c3b[5][17]*mhd->snx[i][j][grid->nzm4]);
				mhd->sny[i][j][grid->nzm1]=grid->c1[5][18]*mhd->sny[i][j][grid->nzm3]+grid->c2[5][18]*mhd->sny[i][j][grid->nz-5]+grid->d[5][18]*(grid->c3a[5][18]*mhd->sny[i][j][grid->nzm2]+grid->c3b[5][18]*mhd->sny[i][j][grid->nzm4]);
				mhd->snz[i][j][grid->nzm1]=grid->c1[5][19]*mhd->snz[i][j][grid->nzm3]+grid->c2[5][19]*mhd->snz[i][j][grid->nz-5]+grid->d[5][19]*(grid->c3a[5][19]*mhd->snz[i][j][grid->nzm2]+grid->c3b[5][19]*mhd->snz[i][j][grid->nzm4]);
				//Check for k=-1
				if(grid->k[5][8]==-1) mhd->bx[i][j][grid->nzm2]=0.;
				if(grid->k[5][9]==-1) mhd->by[i][j][grid->nzm2]=0.;
				if(grid->k[5][10]==-1) mhd->bz[i][j][grid->nzm2]=0.;
				if(grid->k[5][11]==-1) mhd->sx[i][j][grid->nzm2]=0.;
				if(grid->k[5][12]==-1) mhd->sy[i][j][grid->nzm2]=0.;
				if(grid->k[5][13]==-1) mhd->sz[i][j][grid->nzm2]=0.;
				if(grid->k[5][14]==-1) mhd->six[i][j][grid->nzm2]=0.;
				if(grid->k[5][15]==-1) mhd->siy[i][j][grid->nzm2]=0.;
				if(grid->k[5][16]==-1) mhd->siz[i][j][grid->nzm2]=0.;
				if(grid->k[5][17]==-1) mhd->snx[i][j][grid->nzm2]=0.;
				if(grid->k[5][18]==-1) mhd->sny[i][j][grid->nzm2]=0.;
				if(grid->k[5][19]==-1) mhd->snz[i][j][grid->nzm2]=0.;
			}
		}
		}/*-- End of Single Block --*/
	#pragma omp barrier
	}
//	Set div(B)=0 on boundary (note 2->grid->nzm2 so neighboring walls do not contribute)
//	// X-Boundaries
//	for(j=2;j<grid->nym2;j++)
//	{
//		for(k=2;k<grid->nzm2;k++)
//		{
//			mhd->bx[0][j][k]=mhd->bx[2][j][k]+(grid->dify[1][j][k]*(mhd->by[1][j+1][k]-mhd->by[1][j-1][k])+grid->difz[1][j][k]*(mhd->bz[1][j][k+1]-mhd->bz[1][j][k-1]))/grid->difx[1][j][k];
//			mhd->bx[grid->nxm1][j][k]=mhd->bx[grid->nxm3][j][k]-(grid->dify[grid->nxm2][j][k]*(mhd->by[grid->nxm2][j+1][k]-mhd->by[grid->nxm2][j-1][k])+grid->difz[grid->nxm2][j][k]*(mhd->bz[grid->nxm2][j][k+1]-mhd->bz[grid->nxm2][j][k-1]))/grid->difx[grid->nxm2][j][k];
//		}
//	}
//	// Y-Boundaries
//	for(i=2;i<grid->nxm2;i++)
//	{
//		for(k=2;k<grid->nzm2;k++)
//		{
//			mhd->by[i][0][k]=mhd->by[i][2][k]+(grid->difx[i][1][k]*(mhd->bx[i+1][1][k]-mhd->bx[i-1][1][k])+grid->difz[i][1][k]*(mhd->bz[i][1][k+1]-mhd->bz[i][1][k-1]))/grid->dify[i][1][k];
//			mhd->by[i][grid->nym1][k]=mhd->by[i][grid->nym3][k]-(grid->difx[i][grid->nym2][k]*(mhd->bx[i+1][grid->nym2][k]-mhd->bx[i-1][grid->nym2][k])+grid->difz[i][grid->nym2][k]*(mhd->bz[i][grid->nym2][k+1]-mhd->bz[i][grid->nym2][k-1]))/grid->dify[i][grid->nym2][k];
//		}
//	}
//	// Z-Boundaries
//	for(i=2;i<grid->nxm2;i++)
//	{
//		for(j=2;j<grid->nym2;j++)
//		{
//			mhd->bz[i][j][0]=mhd->bz[i][j][2]+(grid->difx[i][j][1]*(mhd->bx[i+1][j][1]-mhd->bx[i-1][j][1])+grid->dify[i][j][1]*(mhd->by[i][j+1][1]-mhd->by[i][j-1][1]))/grid->difz[i][j][1];
//			mhd->bz[i][j][grid->nzm1]=mhd->bz[i][j][grid->nzm3]-(grid->difx[i][j][grid->nzm2]*(mhd->bx[i+1][j][grid->nzm2]-mhd->bx[i-1][j][grid->nzm2])+grid->dify[i][j][grid->nzm2]*(mhd->by[i][j+1][grid->nzm2]-mhd->by[i][j-1][grid->nzm2]))/grid->difz[i][j][grid->nzm2];
//		}
//	}
//	Current in Grid
	#pragma omp for
	for(i=1;i<grid->nxm1;i++)
	{
		for(j=1;j<grid->nym1;j++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				mhd->jx[i][j][k]=((mhd->bz[i][j+1][k]-mhd->bz[i][j-1][k])*grid->dify[i][j][k]-(mhd->by[i][j][k+1]-mhd->by[i][j][k-1])*grid->difz[i][j][k]);
				mhd->jy[i][j][k]=((mhd->bx[i][j][k+1]-mhd->bx[i][j][k-1])*grid->difz[i][j][k]-(mhd->bz[i+1][j][k]-mhd->bz[i-1][j][k])*grid->difx[i][j][k]);
				mhd->jz[i][j][k]=((mhd->by[i+1][j][k]-mhd->by[i-1][j][k])*grid->difx[i][j][k]-(mhd->bx[i][j+1][k]-mhd->bx[i][j-1][k])*grid->dify[i][j][k]);
			}
		}
	}
	// Electron Momentum in grid
	#pragma omp for
	for(i=1;i<grid->nxm1;i++)
	{
		for(j=1;j<grid->nym1;j++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				mhd->sex[i][j][k]=memi*mhd->zi[i][j][k]*mhd->six[i][j][k]-memd*mhd->zd[i][j][k]*mhd->sx[i][j][k]-mhd->me*mhd->jx[i][j][k]/mhd->e;
				mhd->sey[i][j][k]=memi*mhd->zi[i][j][k]*mhd->siy[i][j][k]-memd*mhd->zd[i][j][k]*mhd->sy[i][j][k]-mhd->me*mhd->jy[i][j][k]/mhd->e;
				mhd->sez[i][j][k]=memi*mhd->zi[i][j][k]*mhd->siz[i][j][k]-memd*mhd->zd[i][j][k]*mhd->sz[i][j][k]-mhd->me*mhd->jz[i][j][k]/mhd->e;
			}
		}
	}
//	Now we calculate the electron momentum and current at the boundary.
//      X-BC
	if (grid->perx == 1)
	{
		#pragma omp for
		for(j=1;j<grid->nym1;j++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
//				X-min
				mhd-> sex[0][j][k]=mhd-> sex[grid->nxm3][j][k];
				mhd-> sey[0][j][k]=mhd-> sey[grid->nxm3][j][k];
				mhd-> sez[0][j][k]=mhd-> sez[grid->nxm3][j][k];
				mhd-> jx[0][j][k]=mhd-> jx[grid->nxm3][j][k];
				mhd-> jy[0][j][k]=mhd-> jy[grid->nxm3][j][k];
				mhd-> jz[0][j][k]=mhd-> jz[grid->nxm3][j][k];
//				X-max
				mhd-> sex[grid->nxm1][j][k]=mhd-> sex[2][j][k];
				mhd-> sey[grid->nxm1][j][k]=mhd-> sey[2][j][k];
				mhd-> sez[grid->nxm1][j][k]=mhd-> sez[2][j][k];
				mhd-> jx[grid->nxm1][j][k]=mhd-> jx[2][j][k];
				mhd-> jy[grid->nxm1][j][k]=mhd-> jy[2][j][k];
				mhd-> jz[grid->nxm1][j][k]=mhd-> jz[2][j][k];
			}
		}
	}
	else if (grid->perx == 0)
	{
		#pragma omp for
		for(j=1;j<grid->nym1;j++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				//X-Min
				mhd->jx[0][j][j]=grid->c1[0][11]*mhd->jx[2][j][k]+grid->c2[0][11]*mhd->jx[4][j][k]+grid->d[0][11]*(grid->c3a[0][11]*mhd->jx[1][j][k]+grid->c3b[0][11]*mhd->jx[3][j][k]);
				mhd->jy[0][j][j]=grid->c1[0][11]*mhd->jy[2][j][k]+grid->c2[0][11]*mhd->jy[4][j][k]+grid->d[0][11]*(grid->c3a[0][11]*mhd->jy[1][j][k]+grid->c3b[0][11]*mhd->jy[3][j][k]);
				mhd->jz[0][j][j]=grid->c1[0][11]*mhd->jz[2][j][k]+grid->c2[0][11]*mhd->jz[4][j][k]+grid->d[0][11]*(grid->c3a[0][11]*mhd->jz[1][j][k]+grid->c3b[0][11]*mhd->jz[3][j][k]);
				mhd->sex[0][j][k]=grid->c1[0][11]*mhd->sex[2][j][k]+grid->c2[0][11]*mhd->sex[4][j][k]+grid->d[0][11]*(grid->c3a[0][11]*mhd->sex[1][j][k]+grid->c3b[0][11]*mhd->sex[3][j][k]);
				mhd->sey[0][j][k]=grid->c1[0][12]*mhd->sey[2][j][k]+grid->c2[0][12]*mhd->sey[4][j][k]+grid->d[0][12]*(grid->c3a[0][12]*mhd->sey[1][j][k]+grid->c3b[0][12]*mhd->sey[3][j][k]);
				mhd->sez[0][j][k]=grid->c1[0][13]*mhd->sez[2][j][k]+grid->c2[0][13]*mhd->sez[4][j][k]+grid->d[0][13]*(grid->c3a[0][13]*mhd->sez[1][j][k]+grid->c3b[0][13]*mhd->sez[3][j][k]);
				if(grid->k[0][11]==-1)
				{
					// jx=0 sex=0
					mhd->jx[1][j][k]=0.;
					mhd->sex[1][j][k]=0.;
					// B||=const
					mhd->by[1][j][k]=grid->by0[1][j][k];
					mhd->bz[1][j][k]=grid->bz0[1][j][k];
					// B||=-B||
					mhd->by[0][j][k]=-mhd->by[2][j][k];
					mhd->bz[0][j][k]=-mhd->bz[2][j][k];
				}
				//X-Max
				mhd->jx[grid->nxm1][j][k]=grid->c1[1][11]*mhd->jx[grid->nxm3][j][k]+grid->c2[1][11]*mhd->jx[grid->nx-5][j][k]+grid->d[1][11]*(grid->c3a[1][11]*mhd->jx[grid->nxm2][j][k]+grid->c3b[1][11]*mhd->jx[grid->nxm4][j][k]);
				mhd->jy[grid->nxm1][j][k]=grid->c1[1][12]*mhd->jy[grid->nxm3][j][k]+grid->c2[1][12]*mhd->jy[grid->nx-5][j][k]+grid->d[1][12]*(grid->c3a[1][12]*mhd->jy[grid->nxm2][j][k]+grid->c3b[1][12]*mhd->jy[grid->nxm4][j][k]);
				mhd->jz[grid->nxm1][j][k]=grid->c1[1][13]*mhd->jz[grid->nxm3][j][k]+grid->c2[1][13]*mhd->jz[grid->nx-5][j][k]+grid->d[1][13]*(grid->c3a[1][13]*mhd->jz[grid->nxm2][j][k]+grid->c3b[1][13]*mhd->jz[grid->nxm4][j][k]);
				mhd->sex[grid->nxm1][j][k]=grid->c1[1][11]*mhd->sex[grid->nxm3][j][k]+grid->c2[1][11]*mhd->sex[grid->nx-5][j][k]+grid->d[1][11]*(grid->c3a[1][11]*mhd->sex[grid->nxm2][j][k]+grid->c3b[1][11]*mhd->sex[grid->nxm4][j][k]);
				mhd->sey[grid->nxm1][j][k]=grid->c1[1][12]*mhd->sey[grid->nxm3][j][k]+grid->c2[1][12]*mhd->sey[grid->nx-5][j][k]+grid->d[1][12]*(grid->c3a[1][12]*mhd->sey[grid->nxm2][j][k]+grid->c3b[1][12]*mhd->sey[grid->nxm4][j][k]);
				mhd->sez[grid->nxm1][j][k]=grid->c1[1][13]*mhd->sez[grid->nxm3][j][k]+grid->c2[1][13]*mhd->sez[grid->nx-5][j][k]+grid->d[1][13]*(grid->c3a[1][13]*mhd->sez[grid->nxm2][j][k]+grid->c3b[1][13]*mhd->sez[grid->nxm4][j][k]);
				if(grid->k[1][11]==-1)
				{
					// jx=0 sex=0
					mhd->jx[grid->nxm2][j][k]=0.;
					mhd->sex[grid->nxm2][j][k]=0.;
					// B||=const
					mhd->by[grid->nxm2][j][k]=grid->by0[grid->nxm2][j][k];
					mhd->bz[grid->nxm2][j][k]=grid->bz0[grid->nxm2][j][k];
					// B||=-B||
					mhd->by[grid->nxm1][j][k]=-mhd->by[grid->nxm3][j][k];
					mhd->bz[grid->nxm1][j][k]=-mhd->bz[grid->nxm3][j][k];
				}
			}
		}
	}
//	X Time Dependant BC's
	else if (grid->perx == 2)
	{
	}
//      Y-BC (modified for open BC)
	if (grid->pery == 1)
	{
		#pragma omp for
		for(i=1;i<grid->nxm1;i++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
//				Y-min
				mhd-> sex[i][0][k]=mhd-> sex[i][grid->nym3][k];
				mhd-> sey[i][0][k]=mhd-> sey[i][grid->nym3][k];
				mhd-> sez[i][0][k]=mhd-> sez[i][grid->nym3][k];
				mhd-> jx[i][0][k]=mhd-> jx[i][grid->nym3][k];
				mhd-> jy[i][0][k]=mhd-> jy[i][grid->nym3][k];
				mhd-> jz[i][0][k]=mhd-> jz[i][grid->nym3][k];
//				Y-max
				mhd-> sex[i][grid->nym1][k]=mhd-> sex[i][2][k];
				mhd-> sey[i][grid->nym1][k]=mhd-> sey[i][2][k];
				mhd-> sez[i][grid->nym1][k]=mhd-> sez[i][2][k];
				mhd-> jx[i][grid->nym1][k]=mhd-> jx[i][2][k];
				mhd-> jy[i][grid->nym1][k]=mhd-> jy[i][2][k];
				mhd-> jz[i][grid->nym1][k]=mhd-> jz[i][2][k];
			}
		}
	}
	else if (grid->pery == 0)
	{
		#pragma omp for
		for(i=1;i<grid->nxm1;i++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				//Y-min
				mhd->jx[i][0][k]=grid->c1[2][11]*mhd->jx[i][2][k]+grid->c2[2][11]*mhd->jx[i][4][k]+grid->d[2][11]*(grid->c3a[2][11]*mhd->jx[i][1][k]+grid->c3b[2][11]*mhd->jx[i][3][k]);
				mhd->jy[i][0][k]=grid->c1[2][12]*mhd->jy[i][2][k]+grid->c2[2][12]*mhd->jy[i][4][k]+grid->d[2][12]*(grid->c3a[2][12]*mhd->jy[i][1][k]+grid->c3b[2][12]*mhd->jy[i][3][k]);
				mhd->jz[i][0][k]=grid->c1[2][13]*mhd->jz[i][2][k]+grid->c2[2][13]*mhd->jz[i][4][k]+grid->d[2][13]*(grid->c3a[2][13]*mhd->jz[i][1][k]+grid->c3b[2][13]*mhd->jz[i][3][k]);
				mhd->sex[i][0][k]=grid->c1[2][11]*mhd->sex[i][2][k]+grid->c2[2][11]*mhd->sex[i][4][k]+grid->d[2][11]*(grid->c3a[2][11]*mhd->sex[i][1][k]+grid->c3b[2][11]*mhd->sex[i][3][k]);
				mhd->sey[i][0][k]=grid->c1[2][12]*mhd->sey[i][2][k]+grid->c2[2][12]*mhd->sey[i][4][k]+grid->d[2][12]*(grid->c3a[2][12]*mhd->sey[i][1][k]+grid->c3b[2][12]*mhd->sey[i][3][k]);
				mhd->sez[i][0][k]=grid->c1[2][13]*mhd->sez[i][2][k]+grid->c2[2][13]*mhd->sez[i][4][k]+grid->d[2][13]*(grid->c3a[2][13]*mhd->sez[i][1][k]+grid->c3b[2][13]*mhd->sez[i][3][k]);
				if(grid->k[2][12]==-1)
				{
					// jy=0 sey=0
					mhd->jy[i][1][k]=0.;
					mhd->sey[i][1][k]=0.;
					// B||=const
					mhd->bx[i][1][k]=grid->bx0[i][1][k];
					mhd->bz[i][1][k]=grid->bz0[i][1][k];
					// B||=-B||
					mhd->bx[i][0][k]=-mhd->bx[i][1][k];
					mhd->bz[i][0][k]=-mhd->bz[i][1][k];
				}
				//Y-max
				mhd->jx[i][grid->nym1][k]=grid->c1[3][11]*mhd->jx[i][grid->nym3][k]+grid->c2[3][11]*mhd->jx[i][grid->ny-5][k]+grid->d[3][11]*(grid->c3a[3][11]*mhd->jx[i][grid->nym2][k]+grid->c3b[3][11]*mhd->jx[i][grid->nym4][k]);
				mhd->jy[i][grid->nym1][k]=grid->c1[3][12]*mhd->jy[i][grid->nym3][k]+grid->c2[3][12]*mhd->jy[i][grid->ny-5][k]+grid->d[3][12]*(grid->c3a[3][12]*mhd->jy[i][grid->nym2][k]+grid->c3b[3][12]*mhd->jy[i][grid->nym4][k]);
				mhd->jz[i][grid->nym1][k]=grid->c1[3][13]*mhd->jz[i][grid->nym3][k]+grid->c2[3][13]*mhd->jz[i][grid->ny-5][k]+grid->d[3][13]*(grid->c3a[3][13]*mhd->jz[i][grid->nym2][k]+grid->c3b[3][13]*mhd->jz[i][grid->nym4][k]);
				mhd->sex[i][grid->nym1][k]=grid->c1[3][11]*mhd->sex[i][grid->nym3][k]+grid->c2[3][11]*mhd->sex[i][grid->ny-5][k]+grid->d[3][11]*(grid->c3a[3][11]*mhd->sex[i][grid->nym2][k]+grid->c3b[3][11]*mhd->sex[i][grid->nym4][k]);
				mhd->sey[i][grid->nym1][k]=grid->c1[3][12]*mhd->sey[i][grid->nym3][k]+grid->c2[3][12]*mhd->sey[i][grid->ny-5][k]+grid->d[3][12]*(grid->c3a[3][12]*mhd->sey[i][grid->nym2][k]+grid->c3b[3][12]*mhd->sey[i][grid->nym4][k]);
				mhd->sez[i][grid->nym1][k]=grid->c1[3][13]*mhd->sez[i][grid->nym3][k]+grid->c2[3][13]*mhd->sez[i][grid->ny-5][k]+grid->d[3][13]*(grid->c3a[3][13]*mhd->sez[i][grid->nym2][k]+grid->c3b[3][13]*mhd->sez[i][grid->nym4][k]);
				if(grid->k[3][12]==-1)
				{
					// jy=0 sey=0
					mhd->jy[i][grid->nym2][k]=0.;
					mhd->sey[i][grid->nym2][k]=0.;
					// B||=const
					mhd->bx[i][grid->nym2][k]=grid->bx0[i][grid->nym2][k];
					mhd->bz[i][grid->nym2][k]=grid->bz0[i][grid->nym2][k];
					// B||=-B||
					mhd->bx[i][grid->nym1][k]=-mhd->bx[i][grid->nym3][k];
					mhd->bz[i][grid->nym1][k]=-mhd->bz[i][grid->nym3][k];
				}
				
				
			}
		}
	}
//	Y Time Dependant BC's
	else if (grid->pery == 2)
	{
	}
//      Z-BC
	if (grid->perz == 1)
	{
		#pragma omp for
		for(i=1;i<grid->nxm1;i++)
		{
			for(j=1;j<grid->nym1;j++)
			{
//				Z-Min
				mhd-> sex[i][j][0]=mhd-> sex[i][j][grid->nzm3];
				mhd-> sey[i][j][0]=mhd-> sey[i][j][grid->nzm3];
				mhd-> sez[i][j][0]=mhd-> sez[i][j][grid->nzm3];
				mhd-> jx[i][j][0]=mhd-> jx[i][j][grid->nzm3];
				mhd-> jy[i][j][0]=mhd-> jy[i][j][grid->nzm3];
				mhd-> jz[i][j][0]=mhd-> jz[i][j][grid->nzm3];
//				Z-Max
				mhd-> sex[i][j][grid->nzm1]=mhd-> sex[i][j][2];
				mhd-> sey[i][j][grid->nzm1]=mhd-> sey[i][j][2];
				mhd-> sez[i][j][grid->nzm1]=mhd-> sez[i][j][2];
				mhd-> jx[i][j][grid->nzm1]=mhd-> jx[i][j][2];
				mhd-> jy[i][j][grid->nzm1]=mhd-> jy[i][j][2];
				mhd-> jz[i][j][grid->nzm1]=mhd-> jz[i][j][2];
			}
		}
	}
	else if (grid->perz == 0)
	{
		#pragma omp for
		for(i=1;i<grid->nxm1;i++)
		{
			for(j=1;j<grid->nym1;j++)
			{
				//Z-min
				mhd->jx[i][j][0]=grid->c1[4][11]*mhd->jx[i][j][2]+grid->c2[4][11]*mhd->jx[i][j][4]+grid->d[4][11]*(grid->c3a[4][11]*mhd->jx[i][j][1]+grid->c3b[4][11]*mhd->jx[i][j][3]);
				mhd->jy[i][j][0]=grid->c1[4][12]*mhd->jy[i][j][2]+grid->c2[4][12]*mhd->jy[i][j][4]+grid->d[4][12]*(grid->c3a[4][12]*mhd->jy[i][j][1]+grid->c3b[4][12]*mhd->jy[i][j][3]);
				mhd->jz[i][j][0]=grid->c1[4][13]*mhd->jz[i][j][2]+grid->c2[4][13]*mhd->jz[i][j][4]+grid->d[4][13]*(grid->c3a[4][13]*mhd->jz[i][j][1]+grid->c3b[4][13]*mhd->jz[i][j][3]);
				mhd->sex[i][j][0]=grid->c1[4][11]*mhd->sex[i][j][2]+grid->c2[4][11]*mhd->sex[i][j][4]+grid->d[4][11]*(grid->c3a[4][11]*mhd->sex[i][j][1]+grid->c3b[4][11]*mhd->sex[i][j][3]);
				mhd->sey[i][j][0]=grid->c1[4][12]*mhd->sey[i][j][2]+grid->c2[4][12]*mhd->sey[i][j][4]+grid->d[4][12]*(grid->c3a[4][12]*mhd->sey[i][j][1]+grid->c3b[4][12]*mhd->sey[i][j][3]);
				mhd->sez[i][j][0]=grid->c1[4][13]*mhd->sez[i][j][2]+grid->c2[4][13]*mhd->sez[i][j][4]+grid->d[4][13]*(grid->c3a[4][13]*mhd->sez[i][j][1]+grid->c3b[4][13]*mhd->sez[i][j][3]);
				if(grid->k[4][13]==-1)
				{
					// jz=0 sez=0
					mhd->jz[i][j][1]=0.;
					mhd->sez[i][j][1]=0.;
					// B||=const
					mhd->bx[i][j][1]=grid->bx0[i][j][1];
					mhd->by[i][j][1]=grid->by0[i][j][1];
					// B||=-B||
					mhd->bx[i][j][0]=-mhd->bx[i][j][1];
					mhd->by[i][j][0]=-mhd->by[i][j][1];
				}
				//Z-max
				mhd->jx[i][j][grid->nzm1]=grid->c1[5][11]*mhd->jx[i][j][grid->nzm3]+grid->c2[5][11]*mhd->jx[i][j][grid->nz-5]+grid->d[5][11]*(grid->c3a[5][11]*mhd->jx[i][j][grid->nzm2]+grid->c3b[5][11]*mhd->jx[i][j][grid->nzm4]);
				mhd->jy[i][j][grid->nzm1]=grid->c1[5][12]*mhd->jy[i][j][grid->nzm3]+grid->c2[5][12]*mhd->jy[i][j][grid->nz-5]+grid->d[5][12]*(grid->c3a[5][12]*mhd->jy[i][j][grid->nzm2]+grid->c3b[5][12]*mhd->jy[i][j][grid->nzm4]);
				mhd->jz[i][j][grid->nzm1]=grid->c1[5][13]*mhd->jz[i][j][grid->nzm3]+grid->c2[5][13]*mhd->jz[i][j][grid->nz-5]+grid->d[5][13]*(grid->c3a[5][13]*mhd->jz[i][j][grid->nzm2]+grid->c3b[5][13]*mhd->jz[i][j][grid->nzm4]);
				mhd->sex[i][j][grid->nzm1]=grid->c1[5][11]*mhd->sex[i][j][grid->nzm3]+grid->c2[5][11]*mhd->sex[i][j][grid->nz-5]+grid->d[5][11]*(grid->c3a[5][11]*mhd->sex[i][j][grid->nzm2]+grid->c3b[5][11]*mhd->sex[i][j][grid->nzm4]);
				mhd->sey[i][j][grid->nzm1]=grid->c1[5][12]*mhd->sey[i][j][grid->nzm3]+grid->c2[5][12]*mhd->sey[i][j][grid->nz-5]+grid->d[5][12]*(grid->c3a[5][12]*mhd->sey[i][j][grid->nzm2]+grid->c3b[5][12]*mhd->sey[i][j][grid->nzm4]);
				mhd->sez[i][j][grid->nzm1]=grid->c1[5][13]*mhd->sez[i][j][grid->nzm3]+grid->c2[5][13]*mhd->sez[i][j][grid->nz-5]+grid->d[5][13]*(grid->c3a[5][13]*mhd->sez[i][j][grid->nzm2]+grid->c3b[5][13]*mhd->sez[i][j][grid->nzm4]);
				if(grid->k[5][13]==-1)
				{
					// jz=0 sez=0
					mhd->jz[i][j][grid->nzm2]=0.;
					mhd->sez[i][j][grid->nzm2]=0.;
					// B||=const
					mhd->bx[i][j][grid->nzm2]=grid->bx0[i][j][grid->nzm2];
					mhd->by[i][j][grid->nzm2]=grid->by0[i][j][grid->nzm2];
					// B||=-B||
					mhd->bx[i][j][grid->nzm1]=-mhd->bx[i][j][grid->nzm3];
					mhd->by[i][j][grid->nzm1]=-mhd->by[i][j][grid->nzm3];
				}
				
			}
		}
	}
//	Z Time Dependant BC's
	else if (grid->perz == 2)
	{
		#pragma omp for
		for(i=1;i<grid->nxm1;i++)
		{
			for(j=1;j<grid->nym1;j++)
			{
			}
		}
	}
// Compute Electric Field
	#pragma omp for
	for(i=1;i<grid->nxm1;i++)
	{
		for(j=1;j<grid->nym1;j++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				mhd->ex[i][j][k]=mhd->me*(-(grid->difx[i][j][k]*(mhd->pe[i+1][j][k]-mhd->pe[i-1][j][k]))-mhd->e*(mhd->sey[i][j][k]*mhd->bz[i][j][k]-mhd->sez[i][j][k]*mhd->by[i][j][k])/mhd->me)/mhd->rhoe[i][j][k]+mhd->me/mhd->rhoe[i][j][k]*(mhd->rhoe[i][j][k]*mhd->gravx[i][j][k]-mhd->rho[i][j][k]*mhd->nude[i][j][k]*(mhd->sex[i][j][k]/mhd->rhoe[i][j][k]-mhd->sx[i][j][k]/mhd->rho[i][j][k])-mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*(mhd->sex[i][j][k]/mhd->rhoe[i][j][k]-mhd->six[i][j][k]/mhd->rhoi[i][j][k])-mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*(mhd->sex[i][j][k]/mhd->rhoe[i][j][k]-mhd->snx[i][j][k]/mhd->rhon[i][j][k])+mhd->me/(mhd->mi+mhd->me)*(mhd->ioniz*mhd->snx[i][j][k]-mhd->recom*mhd->rhoi[i][j][k]*mhd->sex[i][j][k]/mhd->me))/mhd->e;
				mhd->ey[i][j][k]=mhd->me*(-(grid->dify[i][j][k]*(mhd->pe[i][j+1][k]-mhd->pe[i][j-1][k]))-mhd->e*(mhd->sez[i][j][k]*mhd->bx[i][j][k]-mhd->sex[i][j][k]*mhd->bz[i][j][k])/mhd->me)/mhd->rhoe[i][j][k]+mhd->me/mhd->rhoe[i][j][k]*(mhd->rhoe[i][j][k]*mhd->gravy[i][j][k]-mhd->rho[i][j][k]*mhd->nude[i][j][k]*(mhd->sey[i][j][k]/mhd->rhoe[i][j][k]-mhd->sy[i][j][k]/mhd->rho[i][j][k])-mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*(mhd->sey[i][j][k]/mhd->rhoe[i][j][k]-mhd->siy[i][j][k]/mhd->rhoi[i][j][k])-mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*(mhd->sey[i][j][k]/mhd->rhoe[i][j][k]-mhd->sny[i][j][k]/mhd->rhon[i][j][k])+mhd->me/(mhd->mi+mhd->me)*(mhd->ioniz*mhd->sny[i][j][k]-mhd->recom*mhd->rhoi[i][j][k]*mhd->sey[i][j][k]/mhd->me))/mhd->e;
				mhd->ez[i][j][k]=mhd->me*(-(grid->difz[i][j][k]*(mhd->pe[i][j][k+1]-mhd->pe[i][j][k-1]))-mhd->e*(mhd->sex[i][j][k]*mhd->by[i][j][k]-mhd->sey[i][j][k]*mhd->bx[i][j][k])/mhd->me)/mhd->rhoe[i][j][k]+mhd->me/mhd->rhoe[i][j][k]*(mhd->rhoe[i][j][k]*mhd->gravz[i][j][k]-mhd->rho[i][j][k]*mhd->nude[i][j][k]*(mhd->sez[i][j][k]/mhd->rhoe[i][j][k]-mhd->sz[i][j][k]/mhd->rho[i][j][k])-mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*(mhd->sez[i][j][k]/mhd->rhoe[i][j][k]-mhd->siz[i][j][k]/mhd->rhoi[i][j][k])-mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*(mhd->sez[i][j][k]/mhd->rhoe[i][j][k]-mhd->snz[i][j][k]/mhd->rhon[i][j][k])+mhd->me/(mhd->mi+mhd->me)*(mhd->ioniz*mhd->snz[i][j][k]-mhd->recom*mhd->rhoi[i][j][k]*mhd->sez[i][j][k]/mhd->me))/mhd->e;
			}
		}
	}
//		#pragma omp barrier
//		if (tid == 0)
//		{
//			printf(".............................%04.2f [s] -----\n",(omp_get_wtime()-stime)/TIMESCALE);
//		}
	} /*-- End of Parallel Block --*/
}
//*****************************************************************************
/*
	Function:	first_step
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		9/26/08
	Inputs:		MHD mhd,GRID grid, MHD omhd
	Outputs:	none
	Purpose:	Performs the initial FTCS step of the integration
			Includes Collisional Terms and Ionization Terms

*/
void first_step(MHD *mhd, GRID *grid,MHD *omhd)
{
	int	i,j,k;
	double	rhovvxx[grid->nx][grid->ny][grid->nz],rhovvxy[grid->nx][grid->ny][grid->nz],rhovvxz[grid->nx][grid->ny][grid->nz];
	double	rhovvyy[grid->nx][grid->ny][grid->nz],rhovvyz[grid->nx][grid->ny][grid->nz],rhovvzz[grid->nx][grid->ny][grid->nz];
	double	vxbx[grid->nx][grid->ny][grid->nz],vxby[grid->nx][grid->ny][grid->nz],vxbz[grid->nx][grid->ny][grid->nz];
	double	vixbx[grid->nx][grid->ny][grid->nz],vixby[grid->nx][grid->ny][grid->nz],vixbz[grid->nx][grid->ny][grid->nz];
	double	jxbx[grid->nx][grid->ny][grid->nz],jxby[grid->nx][grid->ny][grid->nz],jxbz[grid->nx][grid->ny][grid->nz];
	double	qed[grid->nx][grid->ny][grid->nz],qei[grid->nx][grid->ny][grid->nz],qee[grid->nx][grid->ny][grid->nz],qen[grid->nx][grid->ny][grid->nz];
	double	qsex[grid->nx][grid->ny][grid->nz],qsey[grid->nx][grid->ny][grid->nz],qsez[grid->nx][grid->ny][grid->nz];
	double	memd,memi,mdpmi,mdpme,mipme,mdpmn,mipmn,mepmn;
	double	mimime,memime,miinv,meinv;

//	Create some basic helpers (reduces number of divisions calculated)
	memd=mhd->me/mhd->md;
	memi=mhd->me/mhd->mi;
	mdpmi=1./(mhd->md+mhd->mi);
	mdpme=1./(mhd->md+mhd->me);
	mipme=1./(mhd->mi+mhd->me);
	mdpmn=1./(mhd->md+mhd->mn);
	mipmn=1./(mhd->mi+mhd->mn);
	mepmn=1./(mhd->me+mhd->mn);
	mimime=mhd->mi*mipme;
	memime=mhd->me*mipme;
	meinv=1./mhd->me;
	miinv=1./mhd->mi;
//      Initialize helper arrays
	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				rhovvxx[i][j][k]=0.0;
				rhovvxy[i][j][k]=0.0;
				rhovvxz[i][j][k]=0.0;
				rhovvyy[i][j][k]=0.0;
				rhovvyz[i][j][k]=0.0;
				rhovvzz[i][j][k]=0.0;
				vxbx[i][j][k]=0.0;
				vxby[i][j][k]=0.0;
				vxbz[i][j][k]=0.0;
				vixbx[i][j][k]=0.0;
				vixby[i][j][k]=0.0;
				vixbz[i][j][k]=0.0;
				jxbx[i][j][k]=0.0;
				jxby[i][j][k]=0.0;
				jxbz[i][j][k]=0.0;
			}
		}
	}
//      Create previous Timestep
	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				omhd->rho[i][j][k]=mhd->rho[i][j][k];
				omhd->rhoi[i][j][k]=mhd->rhoi[i][j][k];
				omhd->rhoe[i][j][k]=mhd->rhoe[i][j][k];
				omhd->rhon[i][j][k]=mhd->rhon[i][j][k];
				omhd->bx[i][j][k]=mhd->bx[i][j][k];
				omhd->by[i][j][k]=mhd->by[i][j][k];
				omhd->bz[i][j][k]=mhd->bz[i][j][k];
				omhd->sx[i][j][k]=mhd->sx[i][j][k];
				omhd->sy[i][j][k]=mhd->sy[i][j][k];
				omhd->sz[i][j][k]=mhd->sz[i][j][k];
				omhd->six[i][j][k]=mhd->six[i][j][k];
				omhd->siy[i][j][k]=mhd->siy[i][j][k];
				omhd->siz[i][j][k]=mhd->siz[i][j][k];
				omhd->sex[i][j][k]=mhd->sex[i][j][k];
				omhd->sey[i][j][k]=mhd->sey[i][j][k];
				omhd->sez[i][j][k]=mhd->sez[i][j][k];
				omhd->snx[i][j][k]=mhd->snx[i][j][k];
				omhd->sny[i][j][k]=mhd->sny[i][j][k];
				omhd->snz[i][j][k]=mhd->snz[i][j][k];
				omhd->p[i][j][k]=mhd->p[i][j][k];
				omhd->pi[i][j][k]=mhd->pi[i][j][k];
				omhd->pe[i][j][k]=mhd->pe[i][j][k];
				omhd->pn[i][j][k]=mhd->pn[i][j][k];
				omhd->jx[i][j][k]=mhd->jx[i][j][k];
				omhd->jy[i][j][k]=mhd->jy[i][j][k];
				omhd->jz[i][j][k]=mhd->jz[i][j][k];
				omhd->zd[i][j][k]=mhd->zd[i][j][k];
				omhd->zi[i][j][k]=mhd->zi[i][j][k];
				omhd->gamma[i][j][k]=mhd->gamma[i][j][k];
				omhd->gammai[i][j][k]=mhd->gammai[i][j][k];
				omhd->gammae[i][j][k]=mhd->gammae[i][j][k];
				omhd->gamman[i][j][k]=mhd->gamman[i][j][k];
				omhd->nuid[i][j][k]=mhd->nuid[i][j][k];
				omhd->nuie[i][j][k]=mhd->nuie[i][j][k];
				omhd->nude[i][j][k]=mhd->nude[i][j][k];
				omhd->nudn[i][j][k]=mhd->nudn[i][j][k];
				omhd->nuin[i][j][k]=mhd->nuie[i][j][k];
				omhd->nuen[i][j][k]=mhd->nuen[i][j][k];
				omhd->gravx[i][j][k]=mhd->gravx[i][j][k];
				omhd->gravy[i][j][k]=mhd->gravy[i][j][k];
				omhd->gravz[i][j][k]=mhd->gravz[i][j][k];
			}
		}
	}
	omhd->md=mhd->md;
	omhd->mi=mhd->mi;
	omhd->me=mhd->me;
	omhd->mn=mhd->mn;
	omhd->recom=mhd->recom;
	omhd->ioniz=mhd->ioniz;
//      Densities:  Integrate MHD with a FTCS scheme
	for(i=1;i<grid->nxm1;i++)
	{
		for(j=1;j<grid->nym1;j++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				//Densities
				mhd->rho[i][j][k]=omhd->rho[i][j][k]-grid->dt*grid->difx[i][j][k]*(omhd->sx[i+1][j][k]-omhd->sx[i-1][j][k])-grid->dt*grid->dify[i][j][k]*(omhd->sy[i][j+1][k]-omhd->sy[i][j][k])-grid->dt*grid->difz[i][j][k]*(omhd->sz[i][j][k+1]-omhd->sz[i][j][k-1]);
				mhd->rhoi[i][j][k]=omhd->rhoi[i][j][k]-grid->dt*grid->difx[i][j][k]*(omhd->six[i+1][j][k]-omhd->six[i-1][j][k])-grid->dt*grid->dify[i][j][k]*(omhd->siy[i][j+1][k]-omhd->siy[i][j][k])-grid->dt*grid->difz[i][j][k]*(omhd->siz[i][j][k+1]-omhd->siz[i][j][k-1])+grid->dt*(mimime*omhd->ioniz*omhd->rhon[i][j][k]-mimime*omhd->recom*omhd->rhoi[i][j][k]*omhd->rhoe[i][j][k]/omhd->me);
				mhd->rhon[i][j][k]=omhd->rhon[i][j][k]-grid->dt*grid->difx[i][j][k]*(omhd->snx[i+1][j][k]-omhd->snx[i-1][j][k])-grid->dt*grid->dify[i][j][k]*(omhd->sny[i][j+1][k]-omhd->sny[i][j][k])-grid->dt*grid->difz[i][j][k]*(omhd->snz[i][j][k+1]-omhd->snz[i][j][k-1])-grid->dt*(omhd->ioniz*omhd->rhon[i][j][k]-omhd->recom*omhd->rhoi[i][j][k]*omhd->rhoe[i][j][k]/omhd->me);
				if (grid->econt)
				{
					mhd->rhoe[i][j][k]=omhd->rhoe[i][j][k]-grid->dt*grid->difx[i][j][k]*(omhd->sex[i+1][j][k]-omhd->sex[i-1][j][k])-grid->dt*grid->dify[i][j][k]*(omhd->sey[i][j+1][k]-omhd->sey[i][j][k])-grid->dt*grid->difz[i][j][k]*(omhd->sez[i][j][k+1]-omhd->sez[i][j][k-1]);
				}
				else
				{
					mhd->rhoe[i][j][k]=mhd->me*(mhd->zi[i][j][k]*mhd->rhoi[i][j][k]/mhd->mi-mhd->zd[i][j][k]*mhd->rho[i][j][k]);
				}
			}
		}
	}
//	Create rho*v*v,vxb,vixb (For Dust Momentum)
	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				//	rho*v*v
				rhovvxx[i][j][k]=omhd->sx[i][j][k]*omhd->sx[i][j][k]/omhd->rho[i][j][k];
				rhovvxy[i][j][k]=omhd->sx[i][j][k]*omhd->sy[i][j][k]/omhd->rho[i][j][k];
				rhovvxz[i][j][k]=omhd->sx[i][j][k]*omhd->sz[i][j][k]/omhd->rho[i][j][k];
				rhovvyy[i][j][k]=omhd->sy[i][j][k]*omhd->sy[i][j][k]/omhd->rho[i][j][k];
				rhovvyz[i][j][k]=omhd->sy[i][j][k]*omhd->sz[i][j][k]/omhd->rho[i][j][k];
				rhovvzz[i][j][k]=omhd->sz[i][j][k]*omhd->sz[i][j][k]/omhd->rho[i][j][k];
				//	vxb term
				vxbx[i][j][k]=omhd->zd[i][j][k]/omhd->md*(1.+memd*omhd->zd[i][j][k]*omhd->rho[i][j][k]/omhd->rhoe[i][j][k])*(omhd->sy[i][j][k]*omhd->bz[i][j][k]-omhd->sz[i][j][k]*omhd->by[i][j][k]);
				vxby[i][j][k]=omhd->zd[i][j][k]/omhd->md*(1.+memd*omhd->zd[i][j][k]*omhd->rho[i][j][k]/omhd->rhoe[i][j][k])*(omhd->sz[i][j][k]*omhd->bx[i][j][k]-omhd->sx[i][j][k]*omhd->bz[i][j][k]);
				vxbz[i][j][k]=omhd->zd[i][j][k]/omhd->md*(1.+memd*omhd->zd[i][j][k]*omhd->rho[i][j][k]/omhd->rhoe[i][j][k])*(omhd->sx[i][j][k]*omhd->by[i][j][k]-omhd->sy[i][j][k]*omhd->bx[i][j][k]);
				//	vixb term
				vixbx[i][j][k]=omhd->zd[i][j][k]*omhd->zi[i][j][k]*omhd->rho[i][j][k]*memi/(omhd->md*omhd->rhoe[i][j][k])*(omhd->siy[i][j][k]*omhd->bz[i][j][k]-omhd->siz[i][j][k]*omhd->by[i][j][k]);
				vixby[i][j][k]=omhd->zd[i][j][k]*omhd->zi[i][j][k]*omhd->rho[i][j][k]*memi/(omhd->md*omhd->rhoe[i][j][k])*(omhd->siz[i][j][k]*omhd->bx[i][j][k]-omhd->six[i][j][k]*omhd->bz[i][j][k]);
				vixbz[i][j][k]=omhd->zd[i][j][k]*omhd->zi[i][j][k]*omhd->rho[i][j][k]*memi/(omhd->md*omhd->rhoe[i][j][k])*(omhd->six[i][j][k]*omhd->by[i][j][k]-omhd->siy[i][j][k]*omhd->bx[i][j][k]);
				//	jxb term
				jxbx[i][j][k]=omhd->zd[i][j][k]*memd*omhd->rho[i][j][k]/omhd->rhoe[i][j][k]*(omhd->jy[i][j][k]*omhd->bz[i][j][k]-omhd->jz[i][j][k]*omhd->by[i][j][k]);
				jxby[i][j][k]=omhd->zd[i][j][k]*memd*omhd->rho[i][j][k]/omhd->rhoe[i][j][k]*(omhd->jz[i][j][k]*omhd->bx[i][j][k]-omhd->jx[i][j][k]*omhd->bz[i][j][k]);
				jxbz[i][j][k]=omhd->zd[i][j][k]*memd*omhd->rho[i][j][k]/omhd->rhoe[i][j][k]*(omhd->jx[i][j][k]*omhd->by[i][j][k]-omhd->jy[i][j][k]*omhd->bx[i][j][k]);
			}
		}
	}
//	Dust Momentum:  Integrate with FTCS scheme
	for(i=1;i<grid->nxm1;i++)
	{
		for(j=1;j<grid->nym1;j++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				//   d/dk terms
				mhd->sx[i][j][k]=omhd->sx[i][j][k]-grid->dt*grid->difx[i][j][k]*(rhovvxx[i+1][j][k]-rhovvxx[i-1][j][k]+omhd->p[i+1][j][k]-omhd->p[i-1][j][k])-grid->dt*grid->dify[i][j][k]*(rhovvxy[i][j+1][k]-rhovvxy[i][j-1][k])-grid->dt*grid->difz[i][j][k]*(rhovvxz[i][j][k+1]-rhovvxz[i][j][k-1])+omhd->zd[i][j][k]*memd*omhd->rho[i][j][k]/omhd->rhoe[i][j][k]*grid->dt*grid->difx[i][j][k]*(omhd->pe[i+1][j][k]-omhd->pe[i-1][j][k]);
				mhd->sy[i][j][k]=omhd->sy[i][j][k]-grid->dt*grid->dify[i][j][k]*(rhovvyy[i][j+1][k]-rhovvyy[i][j-1][k]+omhd->p[i][j+1][k]-omhd->p[i][j-1][k])-grid->dt*grid->difx[i][j][k]*(rhovvxy[i+1][j][k]-rhovvxy[i-1][j][k])-grid->dt*grid->difz[i][j][k]*(rhovvyz[i][j][k+1]-rhovvyz[i][j][k-1])+omhd->zd[i][j][k]*memd*omhd->rho[i][j][k]/omhd->rhoe[i][j][k]*grid->dt*grid->dify[i][j][k]*(omhd->pe[i][j+1][k]-omhd->pe[i][j-1][k]);
				mhd->sz[i][j][k]=omhd->sz[i][j][k]-grid->dt*grid->difz[i][j][k]*(rhovvzz[i][j][k+1]-rhovvzz[i][j][k-1]+omhd->p[i][j][k+1]-omhd->p[i][j][k-1])-grid->dt*grid->difx[i][j][k]*(rhovvxz[i+1][j][k]-rhovvxz[i-1][j][k])-grid->dt*grid->dify[i][j][k]*(rhovvyz[i][j+1][k]-rhovvyz[i][j-1][k])+omhd->zd[i][j][k]*memd*omhd->rho[i][j][k]/omhd->rhoe[i][j][k]*grid->dt*grid->difz[i][j][k]*(omhd->pe[i][j][k+1]-omhd->pe[i][j][k-1]);
				//   Source Terms
				mhd->sx[i][j][k]=mhd->sx[i][j][k]-grid->dt*(vxbx[i][j][k]-vixbx[i][j][k]+jxbx[i][j][k]-omhd->rho[i][j][k]*omhd->gravx[i][j][k]+omhd->rhoi[i][j][k]*omhd->nuid[i][j][k]*mdpmi*(omhd->sx[i][j][k]/omhd->rho[i][j][k]-omhd->six[i][j][k]/omhd->rhoi[i][j][k])+omhd->rho[i][j][k]*omhd->nude[i][j][k]*mdpme*(omhd->sx[i][j][k]/omhd->rho[i][j][k]-omhd->sex[i][j][k]/omhd->rhoe[i][j][k])+omhd->rho[i][j][k]*omhd->nudn[i][j][k]*mdpmn*(omhd->sx[i][j][k]/omhd->rho[i][j][k]-omhd->snx[i][j][k]/omhd->rhon[i][j][k])+omhd->me*omhd->rho[i][j][k]/omhd->rhoe[i][j][k]*(omhd->rhoe[i][j][k]*omhd->gravx[i][j][k]-omhd->rho[i][j][k]*omhd->nude[i][j][k]*mdpme*(omhd->sex[i][j][k]/omhd->rhoe[i][j][k]-omhd->sx[i][j][k]/omhd->rho[i][j][k])-omhd->rhoi[i][j][k]*omhd->nuie[i][j][k]*mipme*(omhd->sex[i][j][k]/omhd->rhoe[i][j][k]-omhd->six[i][j][k]/omhd->rhoi[i][j][k])-omhd->rhon[i][j][k]*omhd->nuen[i][j][k]*mepmn*(omhd->sex[i][j][k]/omhd->rhoe[i][j][k]-omhd->snx[i][j][k]/omhd->rhon[i][j][k])));
				mhd->sy[i][j][k]=mhd->sy[i][j][k]-grid->dt*(vxby[i][j][k]-vixby[i][j][k]+jxby[i][j][k]-omhd->rho[i][j][k]*omhd->gravy[i][j][k]+omhd->rhoi[i][j][k]*omhd->nuid[i][j][k]*mdpmi*(omhd->sy[i][j][k]/omhd->rho[i][j][k]-omhd->siy[i][j][k]/omhd->rhoi[i][j][k])+omhd->rho[i][j][k]*omhd->nude[i][j][k]*mdpme*(omhd->sy[i][j][k]/omhd->rho[i][j][k]-omhd->sey[i][j][k]/omhd->rhoe[i][j][k])+omhd->rho[i][j][k]*omhd->nudn[i][j][k]*mdpmn*(omhd->sy[i][j][k]/omhd->rho[i][j][k]-omhd->sny[i][j][k]/omhd->rhon[i][j][k])+omhd->me*omhd->rho[i][j][k]/omhd->rhoe[i][j][k]*(omhd->rhoe[i][j][k]*omhd->gravy[i][j][k]-omhd->rho[i][j][k]*omhd->nude[i][j][k]*mdpme*(omhd->sey[i][j][k]/omhd->rhoe[i][j][k]-omhd->sy[i][j][k]/omhd->rho[i][j][k])-omhd->rhoi[i][j][k]*omhd->nuie[i][j][k]*mipme*(omhd->sey[i][j][k]/omhd->rhoe[i][j][k]-omhd->siy[i][j][k]/omhd->rhoi[i][j][k])-omhd->rhon[i][j][k]*omhd->nuen[i][j][k]*mepmn*(omhd->sey[i][j][k]/omhd->rhoe[i][j][k]-omhd->sny[i][j][k]/omhd->rhon[i][j][k])));
				mhd->sz[i][j][k]=mhd->sz[i][j][k]-grid->dt*(vxbz[i][j][k]-vixbz[i][j][k]+jxbz[i][j][k]-omhd->rho[i][j][k]*omhd->gravz[i][j][k]+omhd->rhoi[i][j][k]*omhd->nuid[i][j][k]*mdpmi*(omhd->sz[i][j][k]/omhd->rho[i][j][k]-omhd->siz[i][j][k]/omhd->rhoi[i][j][k])+omhd->rho[i][j][k]*omhd->nude[i][j][k]*mdpme*(omhd->sz[i][j][k]/omhd->rho[i][j][k]-omhd->sez[i][j][k]/omhd->rhoe[i][j][k])+omhd->rho[i][j][k]*omhd->nudn[i][j][k]*mdpmn*(omhd->sz[i][j][k]/omhd->rho[i][j][k]-omhd->snz[i][j][k]/omhd->rhon[i][j][k])+omhd->me*omhd->rho[i][j][k]/omhd->rhoe[i][j][k]*(omhd->rhoe[i][j][k]*omhd->gravz[i][j][k]-omhd->rho[i][j][k]*omhd->nude[i][j][k]*mdpme*(omhd->sez[i][j][k]/omhd->rhoe[i][j][k]-omhd->sz[i][j][k]/omhd->rho[i][j][k])-omhd->rhoi[i][j][k]*omhd->nuie[i][j][k]*mipme*(omhd->sez[i][j][k]/omhd->rhoe[i][j][k]-omhd->siz[i][j][k]/omhd->rhoi[i][j][k])-omhd->rhon[i][j][k]*omhd->nuen[i][j][k]*mepmn*(omhd->sez[i][j][k]/omhd->rhoe[i][j][k]-omhd->snz[i][j][k]/omhd->rhon[i][j][k])));
			}
		}
	}
//	Create rho*v*v,vxb,vixb (For Ion Momentum)
	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				//	rho*v*v
				rhovvxx[i][j][k]=omhd->six[i][j][k]*omhd->six[i][j][k]/omhd->rhoi[i][j][k];
				rhovvxy[i][j][k]=omhd->six[i][j][k]*omhd->siy[i][j][k]/omhd->rhoi[i][j][k];
				rhovvxz[i][j][k]=omhd->six[i][j][k]*omhd->siz[i][j][k]/omhd->rhoi[i][j][k];
				rhovvyy[i][j][k]=omhd->siy[i][j][k]*omhd->siy[i][j][k]/omhd->rhoi[i][j][k];
				rhovvyz[i][j][k]=omhd->siy[i][j][k]*omhd->siz[i][j][k]/omhd->rhoi[i][j][k];
				rhovvzz[i][j][k]=omhd->siz[i][j][k]*omhd->siz[i][j][k]/omhd->rhoi[i][j][k];
				//	vxb term
				vxbx[i][j][k]=omhd->zd[i][j][k]*omhd->zi[i][j][k]*omhd->rhoi[i][j][k]*memd/(omhd->mi*omhd->rhoe[i][j][k])*(omhd->sy[i][j][k]*omhd->bz[i][j][k]-omhd->sz[i][j][k]*omhd->by[i][j][k]);
				vxby[i][j][k]=omhd->zd[i][j][k]*omhd->zi[i][j][k]*omhd->rhoi[i][j][k]*memd/(omhd->mi*omhd->rhoe[i][j][k])*(omhd->sz[i][j][k]*omhd->bx[i][j][k]-omhd->sx[i][j][k]*omhd->bz[i][j][k]);
				vxbz[i][j][k]=omhd->zd[i][j][k]*omhd->zi[i][j][k]*omhd->rhoi[i][j][k]*memd/(omhd->mi*omhd->rhoe[i][j][k])*(omhd->sx[i][j][k]*omhd->by[i][j][k]-omhd->sy[i][j][k]*omhd->bx[i][j][k]);
				//	vixb term
				vixbx[i][j][k]=omhd->zi[i][j][k]/omhd->mi*(1.+memi*omhd->zi[i][j][k]*omhd->rhoi[i][j][k]/omhd->rhoe[i][j][k])*(omhd->siy[i][j][k]*omhd->bz[i][j][k]-omhd->siz[i][j][k]*omhd->by[i][j][k]);
				vixby[i][j][k]=omhd->zi[i][j][k]/omhd->mi*(1.+memi*omhd->zi[i][j][k]*omhd->rhoi[i][j][k]/omhd->rhoe[i][j][k])*(omhd->siz[i][j][k]*omhd->bx[i][j][k]-omhd->six[i][j][k]*omhd->bz[i][j][k]);
				vixbz[i][j][k]=omhd->zi[i][j][k]/omhd->mi*(1.+memi*omhd->zi[i][j][k]*omhd->rhoi[i][j][k]/omhd->rhoe[i][j][k])*(omhd->six[i][j][k]*omhd->by[i][j][k]-omhd->siy[i][j][k]*omhd->bx[i][j][k]);
				//	jxb term
				jxbx[i][j][k]=omhd->zi[i][j][k]*memi*omhd->rhoi[i][j][k]/omhd->rhoe[i][j][k]*(omhd->jy[i][j][k]*omhd->bz[i][j][k]-omhd->jz[i][j][k]*omhd->by[i][j][k]);
				jxby[i][j][k]=omhd->zi[i][j][k]*memi*omhd->rhoi[i][j][k]/omhd->rhoe[i][j][k]*(omhd->jz[i][j][k]*omhd->bx[i][j][k]-omhd->jx[i][j][k]*omhd->bz[i][j][k]);
				jxbz[i][j][k]=omhd->zi[i][j][k]*memi*omhd->rhoi[i][j][k]/omhd->rhoe[i][j][k]*(omhd->jx[i][j][k]*omhd->by[i][j][k]-omhd->jy[i][j][k]*omhd->bx[i][j][k]);
			}
		}
	}
//	Ion Momentum:  Integrate with FTCS scheme
	for(i=1;i<grid->nxm1;i++)
	{
		for(j=1;j<grid->nym1;j++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				//   d/dk terms
				mhd->six[i][j][k]=omhd->six[i][j][k]-grid->dt*grid->difx[i][j][k]*(rhovvxx[i+1][j][k]-rhovvxx[i-1][j][k]+omhd->pi[i+1][j][k]-omhd->pi[i-1][j][k])-grid->dt*grid->dify[i][j][k]*(rhovvxy[i][j+1][k]-rhovvxy[i][j-1][k])-grid->dt*grid->difz[i][j][k]*(rhovvxz[i][j][k+1]-rhovvxz[i][j][k-1])-omhd->zi[i][j][k]*memi*omhd->rhoi[i][j][k]/omhd->rhoe[i][j][k]*grid->dt*grid->difx[i][j][k]*(omhd->pe[i+1][j][k]-omhd->pe[i-1][j][k]);
				mhd->siy[i][j][k]=omhd->siy[i][j][k]-grid->dt*grid->dify[i][j][k]*(rhovvyy[i][j+1][k]-rhovvyy[i][j-1][k]+omhd->pi[i][j+1][k]-omhd->pi[i][j-1][k])-grid->dt*grid->difx[i][j][k]*(rhovvxy[i+1][j][k]-rhovvxy[i-1][j][k])-grid->dt*grid->difz[i][j][k]*(rhovvyz[i][j][k+1]-rhovvyz[i][j][k-1])-omhd->zi[i][j][k]*memi*omhd->rhoi[i][j][k]/omhd->rhoe[i][j][k]*grid->dt*grid->dify[i][j][k]*(omhd->pe[i][j+1][k]-omhd->pe[i][j-1][k]);
				mhd->siz[i][j][k]=omhd->siz[i][j][k]-grid->dt*grid->difz[i][j][k]*(rhovvzz[i][j][k+1]-rhovvzz[i][j][k-1]+omhd->pi[i][j][k+1]-omhd->pi[i][j][k-1])-grid->dt*grid->difx[i][j][k]*(rhovvxz[i+1][j][k]-rhovvxz[i-1][j][k])-grid->dt*grid->dify[i][j][k]*(rhovvyz[i][j+1][k]-rhovvyz[i][j-1][k])-omhd->zi[i][j][k]*memi*omhd->rhoi[i][j][k]/omhd->rhoe[i][j][k]*grid->dt*grid->difz[i][j][k]*(omhd->pe[i][j][k+1]-omhd->pe[i][j][k-1]);
				//   Source Terms
				mhd->six[i][j][k]=mhd->six[i][j][k]+grid->dt*(vxbx[i][j][k]+vixbx[i][j][k]+jxbx[i][j][k]+omhd->rhoi[i][j][k]*omhd->gravx[i][j][k]-omhd->rhoi[i][j][k]*omhd->nuid[i][j][k]*mdpmi*(omhd->six[i][j][k]/omhd->rhoi[i][j][k]-omhd->sx[i][j][k]/omhd->rho[i][j][k])-omhd->rhoi[i][j][k]*omhd->nuie[i][j][k]*mipme*(omhd->six[i][j][k]/omhd->rhoi[i][j][k]-omhd->sex[i][j][k]/omhd->rhoe[i][j][k])-omhd->rhoi[i][j][k]*omhd->nuin[i][j][k]*mipmn*(omhd->six[i][j][k]/omhd->rhoi[i][j][k]-omhd->snx[i][j][k]/omhd->rhon[i][j][k])+mimime*omhd->ioniz*omhd->snx[i][j][k]-mimime*omhd->recom*omhd->rhoe[i][j][k]*omhd->six[i][j][k]*meinv+memi*omhd->rhoi[i][j][k]/omhd->rhoe[i][j][k]*(omhd->rhoe[i][j][k]*omhd->gravx[i][j][k]-omhd->rho[i][j][k]*omhd->nude[i][j][k]*mdpme*(omhd->sex[i][j][k]/omhd->rhoe[i][j][k]-omhd->sx[i][j][k]/omhd->rho[i][j][k])-omhd->rhoi[i][j][k]*omhd->nuie[i][j][k]*mipme*(omhd->sex[i][j][k]/omhd->rhoe[i][j][k]-omhd->six[i][j][k]/omhd->rhoi[i][j][k])-omhd->rhoe[i][j][k]*omhd->nuen[i][j][k]*mepmn*(omhd->sex[i][j][k]/omhd->rhoe[i][j][k]-omhd->snx[i][j][k]/omhd->rhon[i][j][k])+memime*omhd->ioniz*omhd->snx[i][j][k]-mipme*omhd->recom*omhd->rhoi[i][j][k]*omhd->sex[i][j][k]));
				mhd->siy[i][j][k]=mhd->siy[i][j][k]+grid->dt*(vxby[i][j][k]+vixby[i][j][k]+jxby[i][j][k]+omhd->rhoi[i][j][k]*omhd->gravy[i][j][k]-omhd->rhoi[i][j][k]*omhd->nuid[i][j][k]*mdpmi*(omhd->siy[i][j][k]/omhd->rhoi[i][j][k]-omhd->sy[i][j][k]/omhd->rho[i][j][k])-omhd->rhoi[i][j][k]*omhd->nuie[i][j][k]*mipme*(omhd->siy[i][j][k]/omhd->rhoi[i][j][k]-omhd->sey[i][j][k]/omhd->rhoe[i][j][k])-omhd->rhoi[i][j][k]*omhd->nuin[i][j][k]*mipmn*(omhd->siy[i][j][k]/omhd->rhoi[i][j][k]-omhd->sny[i][j][k]/omhd->rhon[i][j][k])+mimime*omhd->ioniz*omhd->sny[i][j][k]-mimime*omhd->recom*omhd->rhoe[i][j][k]*omhd->siy[i][j][k]*meinv+memi*omhd->rhoi[i][j][k]/omhd->rhoe[i][j][k]*(omhd->rhoe[i][j][k]*omhd->gravy[i][j][k]-omhd->rho[i][j][k]*omhd->nude[i][j][k]*mdpme*(omhd->sey[i][j][k]/omhd->rhoe[i][j][k]-omhd->sy[i][j][k]/omhd->rho[i][j][k])-omhd->rhoi[i][j][k]*omhd->nuie[i][j][k]*mipme*(omhd->sey[i][j][k]/omhd->rhoe[i][j][k]-omhd->siy[i][j][k]/omhd->rhoi[i][j][k])-omhd->rhoe[i][j][k]*omhd->nuen[i][j][k]*mepmn*(omhd->sey[i][j][k]/omhd->rhoe[i][j][k]-omhd->sny[i][j][k]/omhd->rhon[i][j][k])+memime*omhd->ioniz*omhd->sny[i][j][k]-mipme*omhd->recom*omhd->rhoi[i][j][k]*omhd->sey[i][j][k]));
				mhd->siz[i][j][k]=mhd->siz[i][j][k]+grid->dt*(vxbz[i][j][k]+vixbz[i][j][k]+jxbz[i][j][k]+omhd->rhoi[i][j][k]*omhd->gravz[i][j][k]-omhd->rhoi[i][j][k]*omhd->nuid[i][j][k]*mdpmi*(omhd->siz[i][j][k]/omhd->rhoi[i][j][k]-omhd->sz[i][j][k]/omhd->rho[i][j][k])-omhd->rhoi[i][j][k]*omhd->nuie[i][j][k]*mipme*(omhd->siz[i][j][k]/omhd->rhoi[i][j][k]-omhd->sez[i][j][k]/omhd->rhoe[i][j][k])-omhd->rhoi[i][j][k]*omhd->nuin[i][j][k]*mipmn*(omhd->siz[i][j][k]/omhd->rhoi[i][j][k]-omhd->snz[i][j][k]/omhd->rhon[i][j][k])+mimime*omhd->ioniz*omhd->snz[i][j][k]-mimime*omhd->recom*omhd->rhoe[i][j][k]*omhd->siz[i][j][k]*meinv+memi*omhd->rhoi[i][j][k]/omhd->rhoe[i][j][k]*(omhd->rhoe[i][j][k]*omhd->gravz[i][j][k]-omhd->rho[i][j][k]*omhd->nude[i][j][k]*mdpme*(omhd->sez[i][j][k]/omhd->rhoe[i][j][k]-omhd->sz[i][j][k]/omhd->rho[i][j][k])-omhd->rhoi[i][j][k]*omhd->nuie[i][j][k]*mipme*(omhd->sez[i][j][k]/omhd->rhoe[i][j][k]-omhd->siz[i][j][k]/omhd->rhoi[i][j][k])-omhd->rhoe[i][j][k]*omhd->nuen[i][j][k]*mepmn*(omhd->sez[i][j][k]/omhd->rhoe[i][j][k]-omhd->snz[i][j][k]/omhd->rhon[i][j][k])+memime*omhd->ioniz*omhd->snz[i][j][k]-mipme*omhd->recom*omhd->rhoi[i][j][k]*omhd->sez[i][j][k]));
			}
		}
	}
//	Create rho*v*v (For Neutral Momentum)
	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				//	rho*v*v
				rhovvxx[i][j][k]=omhd->snx[i][j][k]*omhd->snx[i][j][k]/omhd->rhon[i][j][k];
				rhovvxy[i][j][k]=omhd->snx[i][j][k]*omhd->sny[i][j][k]/omhd->rhon[i][j][k];
				rhovvxz[i][j][k]=omhd->snx[i][j][k]*omhd->snz[i][j][k]/omhd->rhon[i][j][k];
				rhovvyy[i][j][k]=omhd->sny[i][j][k]*omhd->sny[i][j][k]/omhd->rhon[i][j][k];
				rhovvyz[i][j][k]=omhd->sny[i][j][k]*omhd->snz[i][j][k]/omhd->rhon[i][j][k];
				rhovvzz[i][j][k]=omhd->snz[i][j][k]*omhd->snz[i][j][k]/omhd->rhon[i][j][k];
			}
		}
	}
//	Neutral Momentum:  Integrate with FTCS scheme
	for(i=1;i<grid->nxm1;i++)
	{
		for(j=1;j<grid->nym1;j++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				//   d/dk terms
				mhd->snx[i][j][k]=omhd->snx[i][j][k]-grid->dt*grid->difx[i][j][k]*(rhovvxx[i+1][j][k]-rhovvxx[i-1][j][k]+omhd->pn[i+1][j][k]-omhd->pn[i-1][j][k])-grid->dt*grid->dify[i][j][k]*(rhovvxy[i][j+1][k]-rhovvxy[i][j-1][k])-grid->dt*grid->difz[i][j][k]*(rhovvxz[i][j][k+1]-rhovvxz[i][j][k-1]);
				mhd->sny[i][j][k]=omhd->sny[i][j][k]-grid->dt*grid->dify[i][j][k]*(rhovvyy[i][j+1][k]-rhovvyy[i][j-1][k]+omhd->pn[i][j+1][k]-omhd->pn[i][j-1][k])-grid->dt*grid->difx[i][j][k]*(rhovvxy[i+1][j][k]-rhovvxy[i-1][j][k])-grid->dt*grid->difz[i][j][k]*(rhovvyz[i][j][k+1]-rhovvyz[i][j][k-1]);
				mhd->snz[i][j][k]=omhd->snz[i][j][k]-grid->dt*grid->difz[i][j][k]*(rhovvzz[i][j][k+1]-rhovvzz[i][j][k-1]+omhd->pn[i][j][k+1]-omhd->pn[i][j][k-1])-grid->dt*grid->difx[i][j][k]*(rhovvxz[i+1][j][k]-rhovvxz[i-1][j][k])-grid->dt*grid->dify[i][j][k]*(rhovvyz[i][j+1][k]-rhovvyz[i][j-1][k]);
				//   Source Terms
				mhd->snx[i][j][k]=mhd->snx[i][j][k]+grid->dt*(omhd->rhon[i][j][k]*omhd->gravx[i][j][k]-omhd->rhoi[i][j][k]*omhd->nuin[i][j][k]*mipmn*(omhd->snx[i][j][k]/omhd->rhon[i][j][k]-omhd->six[i][j][k]/omhd->rhoi[i][j][k])-omhd->rhoe[i][j][k]*omhd->nuen[i][j][k]*mepmn*(omhd->snx[i][j][k]/omhd->rhon[i][j][k]-omhd->sex[i][j][k]/omhd->rhoe[i][j][k])-omhd->rho[i][j][k]*omhd->nudn[i][j][k]*mdpmn*(omhd->snx[i][j][k]/omhd->rhon[i][j][k]-omhd->sx[i][j][k]/omhd->rho[i][j][k])-omhd->ioniz*omhd->snx[i][j][k]+mimime*omhd->recom*omhd->rhoe[i][j][k]*omhd->six[i][j][k]*meinv+mipme*omhd->recom*omhd->rhoi[i][j][k]*omhd->sex[i][j][k]);
				mhd->sny[i][j][k]=mhd->sny[i][j][k]+grid->dt*(omhd->rhon[i][j][k]*omhd->gravy[i][j][k]-omhd->rhoi[i][j][k]*omhd->nuin[i][j][k]*mipmn*(omhd->sny[i][j][k]/omhd->rhon[i][j][k]-omhd->siy[i][j][k]/omhd->rhoi[i][j][k])-omhd->rhoe[i][j][k]*omhd->nuen[i][j][k]*mepmn*(omhd->sny[i][j][k]/omhd->rhon[i][j][k]-omhd->sey[i][j][k]/omhd->rhoe[i][j][k])-omhd->rho[i][j][k]*omhd->nudn[i][j][k]*mdpmn*(omhd->sny[i][j][k]/omhd->rhon[i][j][k]-omhd->sy[i][j][k]/omhd->rho[i][j][k])-omhd->ioniz*omhd->sny[i][j][k]+mimime*omhd->recom*omhd->rhoe[i][j][k]*omhd->siy[i][j][k]*meinv+mipme*omhd->recom*omhd->rhoi[i][j][k]*omhd->sey[i][j][k]);
				mhd->snz[i][j][k]=mhd->snz[i][j][k]+grid->dt*(omhd->rhon[i][j][k]*omhd->gravz[i][j][k]-omhd->rhoi[i][j][k]*omhd->nuin[i][j][k]*mipmn*(omhd->snz[i][j][k]/omhd->rhon[i][j][k]-omhd->siz[i][j][k]/omhd->rhoi[i][j][k])-omhd->rhoe[i][j][k]*omhd->nuen[i][j][k]*mepmn*(omhd->snz[i][j][k]/omhd->rhon[i][j][k]-omhd->sez[i][j][k]/omhd->rhoe[i][j][k])-omhd->rho[i][j][k]*omhd->nudn[i][j][k]*mdpmn*(omhd->snz[i][j][k]/omhd->rhon[i][j][k]-omhd->sz[i][j][k]/omhd->rho[i][j][k])-omhd->ioniz*omhd->snz[i][j][k]+mimime*omhd->recom*omhd->rhoe[i][j][k]*omhd->siz[i][j][k]*meinv+mipme*omhd->recom*omhd->rhoi[i][j][k]*omhd->sez[i][j][k]);
			}
		}
	}
//	Create vxb,vixb,jxb (For Induction Equation)
	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				//  vxb term
				vxbx[i][j][k]=(omhd->sy[i][j][k]*omhd->bz[i][j][k]-omhd->by[i][j][k]*omhd->sz[i][j][k])/omhd->rhoe[i][j][k];
				vxby[i][j][k]=(omhd->bx[i][j][k]*omhd->sz[i][j][k]-omhd->sx[i][j][k]*omhd->bz[i][j][k])/omhd->rhoe[i][j][k];
				vxbz[i][j][k]=(omhd->sx[i][j][k]*omhd->by[i][j][k]-omhd->sy[i][j][k]*omhd->bx[i][j][k])/omhd->rhoe[i][j][k];
				//  vixb term
				vixbx[i][j][k]=(omhd->siy[i][j][k]*omhd->bz[i][j][k]-omhd->by[i][j][k]*omhd->siz[i][j][k])/omhd->rhoe[i][j][k];
				vixby[i][j][k]=(omhd->bx[i][j][k]*omhd->siz[i][j][k]-omhd->six[i][j][k]*omhd->bz[i][j][k])/omhd->rhoe[i][j][k];
				vixbz[i][j][k]=(omhd->six[i][j][k]*omhd->by[i][j][k]-omhd->siy[i][j][k]*omhd->bx[i][j][k])/omhd->rhoe[i][j][k];
				//  jxb term
				jxbx[i][j][k]=(omhd->jy[i][j][k]*omhd->bz[i][j][k]-omhd->by[i][j][k]*omhd->jz[i][j][k])/omhd->rhoe[i][j][k];
				jxby[i][j][k]=(omhd->jz[i][j][k]*omhd->bx[i][j][k]-omhd->jx[i][j][k]*omhd->bz[i][j][k])/omhd->rhoe[i][j][k];
				jxbx[i][j][k]=(omhd->jx[i][j][k]*omhd->by[i][j][k]-omhd->jy[i][j][k]*omhd->bx[i][j][k])/omhd->rhoe[i][j][k];
				//  Electron Momentum Source Term Qse
				qsex[i][j][k]=-1.*omhd->rho[i][j][k]*omhd->nude[i][j][k]*mdpme*(omhd->sex[i][j][k]/omhd->rhoe[i][j][k]-omhd->sx[i][j][k]/omhd->rho[i][j][k])-omhd->rhoi[i][j][k]*omhd->nuie[i][j][k]*mipme*(omhd->sex[i][j][k]/omhd->rhoe[i][j][k]-omhd->six[i][j][k]/omhd->rhoi[i][j][k])-omhd->rhoe[i][j][k]*omhd->nuen[i][j][k]*mipme*(omhd->sex[i][j][k]/omhd->rhoe[i][j][k]-omhd->snx[i][j][k]/omhd->rhon[i][j][k])+memime*omhd->ioniz*omhd->snx[i][j][k]-mipme*omhd->recom*omhd->rhoe[i][j][k]*omhd->six[i][j][k];
				qsey[i][j][k]=-1.*omhd->rho[i][j][k]*omhd->nude[i][j][k]*mdpme*(omhd->sey[i][j][k]/omhd->rhoe[i][j][k]-omhd->sy[i][j][k]/omhd->rho[i][j][k])-omhd->rhoi[i][j][k]*omhd->nuie[i][j][k]*mipme*(omhd->sey[i][j][k]/omhd->rhoe[i][j][k]-omhd->siy[i][j][k]/omhd->rhoi[i][j][k])-omhd->rhoe[i][j][k]*omhd->nuen[i][j][k]*mipme*(omhd->sey[i][j][k]/omhd->rhoe[i][j][k]-omhd->sny[i][j][k]/omhd->rhon[i][j][k])+memime*omhd->ioniz*omhd->sny[i][j][k]-mipme*omhd->recom*omhd->rhoe[i][j][k]*omhd->siy[i][j][k];
				qsez[i][j][k]=-1.*omhd->rho[i][j][k]*omhd->nude[i][j][k]*mdpme*(omhd->sez[i][j][k]/omhd->rhoe[i][j][k]-omhd->sz[i][j][k]/omhd->rho[i][j][k])-omhd->rhoi[i][j][k]*omhd->nuie[i][j][k]*mipme*(omhd->sez[i][j][k]/omhd->rhoe[i][j][k]-omhd->siz[i][j][k]/omhd->rhoi[i][j][k])-omhd->rhoe[i][j][k]*omhd->nuen[i][j][k]*mipme*(omhd->sez[i][j][k]/omhd->rhoe[i][j][k]-omhd->snz[i][j][k]/omhd->rhon[i][j][k])+memime*omhd->ioniz*omhd->snz[i][j][k]-mipme*omhd->recom*omhd->rhoe[i][j][k]*omhd->siz[i][j][k];
			}
		}
	}
//	Induction Equations:  Integrate with FTCS scheme
	for(i=1;i<grid->nxm1;i++)
	{
		for(j=1;j<grid->nym1;j++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				//  d/dk term
				mhd->bx[i][j][k]=omhd->bx[i][j][k]-omhd->zd[i][j][k]*memd*grid->dt*(grid->dify[i][j][k]*(vxbz[i][j+1][k]-vxbz[i][j-1][k])+grid->difz[i][j][k]*(vxby[i][j][k+1]-vxby[i][j][k-1]))+omhd->zi[i][j][k]*memi*grid->dt*(grid->dify[i][j][k]*(vixbz[i][j+1][k]-vixbz[i][j-1][k])+grid->difz[i][j][k]*(vixby[i][j][k+1]-vixby[i][j][k-1]))-omhd->me*grid->dt*(grid->dify[i][j][k]*(jxbz[i][j+1][k]-jxbz[i][j-1][k])-grid->difz[i][j][k]*(jxby[i][j][k+1]-jxby[i][j][k-1]))+omhd->me/(omhd->rhoe[i][j][k]*omhd->rhoe[i][j][k])*grid->dt*(grid->dify[i][j][k]*(omhd->pe[i][j+1][k]-omhd->pe[i][j-1][k])*grid->difz[i][j][k]*(omhd->rhoe[i][j][k+1]-omhd->rhoe[i][j][k-1])-grid->dify[i][j][k]*(omhd->rhoe[i][j+1][k]-omhd->rhoe[i][j-1][k])*grid->difz[i][j][k]*(omhd->pe[i][j][k+1]-omhd->pe[i][j][k-1]))-omhd->me*grid->dt*(grid->dify[i][j][k]*(omhd->gravz[i][j+1][k]-omhd->gravz[i][j-1][k]+qsez[i][j+1][k]-qsez[i][j-1][k])-grid->difz[i][j][k]*(omhd->gravy[i][j][k+1]-omhd->gravy[i][j][k-1]+qsey[i][j][k+1]-qsey[i][j][k-1]));
				mhd->by[i][j][k]=omhd->by[i][j][k]-omhd->zd[i][j][k]*memd*grid->dt*(grid->difz[i][j][k]*(vxbx[i][j][k+1]-vxbx[i][j][k-1])-grid->difx[i][j][k]*(vxbz[i+1][j][k])-vxbz[i-1][j][k])+omhd->zi[i][j][k]*memi*grid->dt*(grid->difz[i][j][k]*(vixbx[i][j][k+1]-vixbx[i][j][k-1])-grid->difx[i][j][k]*(vixbz[i+1][j][k])-vixbz[i-1][j][k])-omhd->me*grid->dt*(grid->difz[i][j][k]*(jxbx[i][j][k+1]-jxbx[i][j][k-1])-grid->difx[i][j][k]*(jxbz[i+1][j][k])-jxbz[i-1][j][k])+omhd->me/(omhd->rhoe[i][j][k]*omhd->rhoe[i][j][k])*grid->dt*(grid->difx[i][j][k]*(omhd->rhoe[i+1][j][k]-omhd->rhoe[i-1][j][k])*grid->difz[i][j][k]*(omhd->pe[i][j][k+1]-omhd->pe[i][j][k-1])-grid->difx[i][j][k]*(omhd->pe[i+1][j][k]-omhd->pe[i-1][j][k])*grid->difz[i][j][k]*(omhd->rhoe[i][j][k+1]-omhd->rhoe[i][j][k-1]))-omhd->me*grid->dt*(grid->difz[i][j][k]*(omhd->gravx[i][j][k+1]-omhd->gravx[i][j][k-1]+qsex[i][j][k+1]-qsex[i][j][k-1])-grid->difx[i][j][k]*(omhd->gravz[i+1][j][k]-omhd->gravz[i-1][j][k]+qsez[i+1][j][k]-qsez[i-1][j][k]));
				mhd->bz[i][j][k]=omhd->bz[i][j][k]-omhd->zd[i][j][k]*memd*grid->dt*(grid->difx[i][j][k]*(vxby[i+1][j][k]-vxby[i-1][j][k])-grid->dify[i][j][k]*(vxbx[i][j+1][k]-vxbx[i][j-1][k]))+omhd->zi[i][j][k]*memi*grid->dt*(grid->difx[i][j][k]*(vixby[i+1][j][k]-vixby[i-1][j][k])-grid->dify[i][j][k]*(vixbx[i][j+1][k]-vixbx[i][j-1][k]))-omhd->me*grid->dt*(grid->difx[i][j][k]*(jxby[i+1][j][k]-jxby[i-1][j][k])-grid->dify[i][j][k]*(jxbx[i][j+1][k]-jxbx[i][j-1][k]))+omhd->me/(omhd->rhoe[i][j][k]*omhd->rhoe[i][j][k])*grid->dt*(grid->difx[i][j][k]*(omhd->pe[i+1][j][k]-omhd->pe[i-1][j][k])*grid->dify[i][j][k]*(omhd->rhoe[i][j+1][k]-omhd->rhoe[i][j-1][k])-grid->difx[i][j][k]*(omhd->rhoe[i+1][j][k]-omhd->rhoe[i-1][j][k])*grid->dify[i][j][k]*(omhd->pe[i][j+1][k]-omhd->pe[i][j-1][k]))-omhd->me*grid->dt*(grid->difx[i][j][k]*(omhd->gravy[i+1][j][k]-omhd->gravy[i-1][j][k]+qsey[i+1][j][k]-qsey[i-1][j][k])-grid->dify[i][j][k]*(omhd->gravx[i][j+1][k]-omhd->gravx[i][j-1][k]+qsex[i][j+1][k]-qsex[i][j-1][k]));
			}
		}
	}
//	Create Momentum source terms for Energy Equation Qe
	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				qed[i][j][k]=omhd->rhoi[i][j][k]*omhd->nuid[i][j][k]*mdpmi*(pow(omhd->six[i][j][k]/omhd->rhoi[i][j][k]-omhd->sx[i][j][k]/omhd->rho[i][j][k],2)+pow(omhd->siy[i][j][k]/omhd->rhoi[i][j][k]-omhd->sy[i][j][k]/omhd->rho[i][j][k],2)+pow(omhd->siz[i][j][k]/omhd->rhoi[i][j][k]-omhd->sz[i][j][k]/omhd->rho[i][j][k],2))+omhd->rho[i][j][k]*omhd->nude[i][j][k]*mdpme*(pow(omhd->sex[i][j][k]/omhd->rhoe[i][j][k]-omhd->sx[i][j][k]/omhd->rho[i][j][k],2)+pow(omhd->sey[i][j][k]/omhd->rhoe[i][j][k]-omhd->sy[i][j][k]/omhd->rho[i][j][k],2)+pow(omhd->sez[i][j][k]/omhd->rhoe[i][j][k]-omhd->sz[i][j][k]/omhd->rho[i][j][k],2))+omhd->rho[i][j][k]*omhd->nudn[i][j][k]*mdpmn*(pow(omhd->snx[i][j][k]/omhd->rhon[i][j][k]-omhd->sx[i][j][k]/omhd->rho[i][j][k],2)+pow(omhd->sny[i][j][k]/omhd->rhon[i][j][k]-omhd->sy[i][j][k]/omhd->rho[i][j][k],2)+pow(omhd->snz[i][j][k]/omhd->rhon[i][j][k]-omhd->sz[i][j][k]/omhd->rho[i][j][k],2))-2.*omhd->rhoi[i][j][k]*omhd->nuid[i][j][k]*mdpmi*(omhd->p[i][j][k]/(omhd->gamma[i][j][k]-1.)/omhd->rho[i][j][k]-omhd->mi*omhd->pi[i][j][k]/(omhd->gammai[i][j][k]-1.)/omhd->rhoi[i][j][k])-2.*omhd->rho[i][j][k]*omhd->nude[i][j][k]*mdpme*(omhd->p[i][j][k]/(omhd->gamma[i][j][k]-1.)/omhd->rho[i][j][k]-omhd->me*omhd->pe[i][j][k]/(omhd->gammae[i][j][k]-1.)/omhd->rhoe[i][j][k])-2.*omhd->rho[i][j][k]*omhd->nudn[i][j][k]*mdpmn*(omhd->p[i][j][k]/(omhd->gamma[i][j][k]-1.)/omhd->rho[i][j][k]-omhd->mn*omhd->pn[i][j][k]/(omhd->gamman[i][j][k]-1.)/omhd->rhon[i][j][k]);
				qei[i][j][k]=omhd->rhoi[i][j][k]*(omhd->nuid[i][j][k]*mdpmi*(pow(omhd->sx[i][j][k]/omhd->rho[i][j][k]-omhd->six[i][j][k]/omhd->rhoi[i][j][k],2)+pow(omhd->sy[i][j][k]/omhd->rho[i][j][k]-omhd->siy[i][j][k]/omhd->rhoi[i][j][k],2)+pow(omhd->sz[i][j][k]/omhd->rho[i][j][k]-omhd->siz[i][j][k]/omhd->rhoi[i][j][k],2))+omhd->me*omhd->nuie[i][j][k]*mipme*(pow(omhd->sex[i][j][k]/omhd->rhoe[i][j][k]-omhd->six[i][j][k]/omhd->rhoi[i][j][k],2)+pow(omhd->sey[i][j][k]/omhd->rhoe[i][j][k]-omhd->siy[i][j][k]/omhd->rhoi[i][j][k],2)+pow(omhd->sez[i][j][k]/omhd->rhoe[i][j][k]-omhd->siz[i][j][k]/omhd->rhoi[i][j][k],2))+omhd->nuin[i][j][k]*mipmn*(pow(omhd->snx[i][j][k]/omhd->rhon[i][j][k]-omhd->six[i][j][k]/omhd->rhoi[i][j][k],2)+pow(omhd->sny[i][j][k]/omhd->rhon[i][j][k]-omhd->siy[i][j][k]/omhd->rhoi[i][j][k],2)+pow(omhd->snz[i][j][k]/omhd->rhon[i][j][k]-omhd->siz[i][j][k]/omhd->rhoi[i][j][k],2)))-2.*omhd->rhoi[i][j][k]*omhd->nuid[i][j][k]*mdpmi*(omhd->mi*omhd->pi[i][j][k]/(omhd->gammai[i][j][k]-1.)/omhd->rhoi[i][j][k]-omhd->p[i][j][k]/(omhd->gamma[i][j][k]-1.)/omhd->rho[i][j][k])-2.*omhd->rhoi[i][j][k]*omhd->nuie[i][j][k]*mipme*(omhd->mi*omhd->pi[i][j][k]/(omhd->gammai[i][j][k]-1.)/omhd->rhoi[i][j][k]-omhd->me*omhd->pe[i][j][k]/(omhd->gammae[i][j][k]-1.)/omhd->rhoe[i][j][k])-2.*omhd->rhoi[i][j][k]*omhd->nuin[i][j][k]*mipmn*(omhd->mi*omhd->pi[i][j][k]/(omhd->gammai[i][j][k]-1.)/omhd->rhoi[i][j][k]-omhd->mn*omhd->pn[i][j][k]/(omhd->gamman[i][j][k]-1.)/omhd->rhon[i][j][k])+omhd->gamman[i][j][k]*mipme*omhd->pn[i][j][k]*omhd->ioniz*omhd->mn/(omhd->gamman[i][j][k]-1.)-omhd->gammai[i][j][k]*omhd->pi[i][j][k]*omhd->recom*omhd->rhoe[i][j][k]*meinv/(omhd->gammai[i][j][k]-1.)+0.25*mimime*(omhd->ioniz*omhd->rhon[i][j][k]+omhd->recom*omhd->rhoi[i][j][k]*omhd->rhoe[i][j][k]*meinv)*(pow(omhd->snx[i][j][k]/omhd->rhon[i][j][k]-omhd->six[i][j][k]/omhd->rhoi[i][j][k],2)+pow(omhd->sny[i][j][k]/omhd->rhon[i][j][k]-omhd->siy[i][j][k]/omhd->rhoi[i][j][k],2)+pow(omhd->snz[i][j][k]/omhd->rhon[i][j][k]-omhd->siz[i][j][k]/omhd->rhoi[i][j][k],2));
				qee[i][j][k]=omhd->rho[i][j][k]*omhd->nude[i][j][k]*mdpme*(pow(omhd->sx[i][j][k]/omhd->rho[i][j][k]-omhd->sex[i][j][k]/omhd->rhoe[i][j][k],2)+pow(omhd->sy[i][j][k]/omhd->rho[i][j][k]-omhd->sey[i][j][k]/omhd->rhoe[i][j][k],2)+pow(omhd->sz[i][j][k]/omhd->rho[i][j][k]-omhd->sez[i][j][k]/omhd->rhoe[i][j][k],2))+omhd->rhoi[i][j][k]*omhd->mi*omhd->nuie[i][j][k]*mipme*(pow(omhd->six[i][j][k]/omhd->rhoi[i][j][k]-omhd->sex[i][j][k]/omhd->rhoe[i][j][k],2)+pow(omhd->siy[i][j][k]/omhd->rhoi[i][j][k]-omhd->sey[i][j][k]/omhd->rhoe[i][j][k],2)+pow(omhd->siz[i][j][k]/omhd->rhoi[i][j][k]-omhd->sez[i][j][k]/omhd->rhoe[i][j][k],2))+omhd->rhoe[i][j][k]*omhd->mi*omhd->nuen[i][j][k]*mipme*(pow(omhd->snx[i][j][k]/omhd->rhon[i][j][k]-omhd->sex[i][j][k]/omhd->rhoe[i][j][k],2)+pow(omhd->sny[i][j][k]/omhd->rhon[i][j][k]-omhd->sey[i][j][k]/omhd->rhoe[i][j][k],2)+pow(omhd->snz[i][j][k]/omhd->rhon[i][j][k]-omhd->sez[i][j][k]/omhd->rhoe[i][j][k],2))-2.*omhd->rho[i][j][k]*omhd->nude[i][j][k]*mdpme*(omhd->me*omhd->pe[i][j][k]/(omhd->gammae[i][j][k]-1.)/omhd->rhoe[i][j][k]-omhd->p[i][j][k]/(omhd->gamma[i][j][k]-1.)/omhd->rho[i][j][k])-2.*omhd->rhoi[i][j][k]*omhd->nuie[i][j][k]*mipme*(omhd->me*omhd->pe[i][j][k]/(omhd->gammae[i][j][k]-1.)/omhd->rhoe[i][j][k]-omhd->mi*omhd->pi[i][j][k]/(omhd->gammai[i][j][k]-1.)/omhd->rhoi[i][j][k])-2.*omhd->rhoe[i][j][k]*omhd->nuen[i][j][k]*mepmn*(omhd->me*omhd->pe[i][j][k]/(omhd->gammae[i][j][k]-1.)/omhd->rhoe[i][j][k]-omhd->mn*omhd->pn[i][j][k]/(omhd->gamman[i][j][k]-1.)/omhd->rhon[i][j][k])+omhd->gamman[i][j][k]*memime*omhd->pn[i][j][k]*omhd->ioniz*omhd->mn*miinv/(omhd->gamman[i][j][k]-1.)-omhd->gammae[i][j][k]*omhd->pe[i][j][k]*omhd->recom*omhd->rhoi[i][j][k]*miinv/(omhd->gammae[i][j][k]-1.)+0.25*memime*(omhd->ioniz*omhd->rhon[i][j][k]+omhd->recom*omhd->rhoi[i][j][k]*omhd->rhoe[i][j][k]*meinv)*(pow(omhd->snx[i][j][k]/omhd->rhon[i][j][k]-omhd->sex[i][j][k]/omhd->rhoe[i][j][k],2)+pow(omhd->sny[i][j][k]/omhd->rhon[i][j][k]-omhd->sey[i][j][k]/omhd->rhoe[i][j][k],2)+pow(omhd->snz[i][j][k]/omhd->rhon[i][j][k]-omhd->sez[i][j][k]/omhd->rhoe[i][j][k],2));
				qen[i][j][k]=omhd->rho[i][j][k]*omhd->nudn[i][j][k]*mdpmn*(pow(omhd->snx[i][j][k]/omhd->rhon[i][j][k]-omhd->sx[i][j][k]/omhd->rho[i][j][k],2)+pow(omhd->sny[i][j][k]/omhd->rhon[i][j][k]-omhd->sy[i][j][k]/omhd->rho[i][j][k],2)+pow(omhd->snz[i][j][k]/omhd->rhon[i][j][k]-omhd->sz[i][j][k]/omhd->rho[i][j][k],2))+omhd->rhoi[i][j][k]*omhd->nuin[i][j][k]*mipmn*(pow(omhd->snx[i][j][k]/omhd->rhon[i][j][k]-omhd->six[i][j][k]/omhd->rhoi[i][j][k],2)+pow(omhd->sny[i][j][k]/omhd->rhon[i][j][k]-omhd->siy[i][j][k]/omhd->rhoi[i][j][k],2)+pow(omhd->snz[i][j][k]/omhd->rhon[i][j][k]-omhd->siz[i][j][k]/omhd->rhoi[i][j][k],2))+omhd->rhoe[i][j][k]*omhd->nuen[i][j][k]*mepmn*(pow(omhd->snx[i][j][k]/omhd->rhon[i][j][k]-omhd->sex[i][j][k]/omhd->rhoe[i][j][k],2)+pow(omhd->sny[i][j][k]/omhd->rhon[i][j][k]-omhd->sey[i][j][k]/omhd->rhoe[i][j][k],2)+pow(omhd->snz[i][j][k]/omhd->rhon[i][j][k]-omhd->sez[i][j][k]/omhd->rhoe[i][j][k],2))-2.*omhd->rho[i][j][k]*omhd->nudn[i][j][k]*mdpmn*(omhd->mn*omhd->pn[i][j][k]/(omhd->gamman[i][j][k]-1.)/omhd->rhon[i][j][k]-omhd->p[i][j][k]/(omhd->gamma[i][j][k]-1.)/omhd->rho[i][j][k])-2.*omhd->rhoi[i][j][k]*omhd->nuin[i][j][k]*mipmn*(omhd->mn*omhd->pn[i][j][k]/(omhd->gamman[i][j][k]-1.)/omhd->rhon[i][j][k]-omhd->mi*omhd->pi[i][j][k]/(omhd->gammai[i][j][k]-1.)/omhd->rhoi[i][j][k])-2.*omhd->rhoe[i][j][k]*omhd->nuen[i][j][k]*mepmn*(omhd->mn*omhd->pn[i][j][k]/(omhd->gamman[i][j][k]-1.)/omhd->rhon[i][j][k]-omhd->me*omhd->pe[i][j][k]/(omhd->gammae[i][j][k]-1.)/omhd->rhoe[i][j][k])-omhd->gamman[i][j][k]*omhd->pn[i][j][k]*omhd->ioniz/(omhd->gamman[i][j][k]-1.)+(omhd->gammae[i][j][k]*omhd->pe[i][j][k]*omhd->me/omhd->rhoe[i][j][k]/(omhd->gammae[i][j][k]-1.)+omhd->gammai[i][j][k]*omhd->pi[i][j][k]*omhd->mi/omhd->rhoi[i][j][k]/(omhd->gammai[i][j][k]-1.))*omhd->recom*omhd->rhoi[i][j][k]*omhd->rhoe[i][j][k]*miinv*meinv+0.25*mimime*(omhd->ioniz*omhd->rhon[i][j][k]+omhd->recom*omhd->rhoi[i][j][k]*omhd->rhoe[i][j][k]*meinv)*(pow(omhd->snx[i][j][k]/omhd->rhon[i][j][k]-omhd->six[i][j][k]/omhd->rhoi[i][j][k],2)+pow(omhd->sny[i][j][k]/omhd->rhon[i][j][k]-omhd->siy[i][j][k]/omhd->rhoi[i][j][k],2)+pow(omhd->snz[i][j][k]/omhd->rhon[i][j][k]-omhd->siz[i][j][k]/omhd->rhoi[i][j][k],2))+0.25*memime*(omhd->ioniz*omhd->rhon[i][j][k]+omhd->recom*omhd->rhoi[i][j][k]*omhd->rhoe[i][j][k]*meinv)*(pow(omhd->snx[i][j][k]/omhd->rhon[i][j][k]-omhd->sex[i][j][k]/omhd->rhoe[i][j][k],2)+pow(omhd->sny[i][j][k]/omhd->rhon[i][j][k]-omhd->sey[i][j][k]/omhd->rhoe[i][j][k],2)+pow(omhd->snz[i][j][k]/omhd->rhon[i][j][k]-omhd->sez[i][j][k]/omhd->rhoe[i][j][k],2));
			}
		}
	}
//	Energy Equations:  Integrate with FTCS Scheme
	for(i=1;i<grid->nxm1;i++)
	{
		for(j=1;j<grid->nym1;j++)
		{
			for(k=1;k<grid->nzm1;k++)
			{
				mhd->p[i][j][k]=omhd->p[i][j][k]-grid->dt*(grid->difx[i][j][k]*(omhd->p[i+1][j][k]*omhd->sx[i+1][j][k]/omhd->rho[i+1][j][k]-omhd->p[i-1][j][k]*omhd->sx[i-1][j][k]/omhd->rho[i-1][j][k])+grid->dify[i][j][k]*(omhd->p[i][j+1][k]*omhd->sy[i][j+1][k]/omhd->rho[i][j+1][k]-omhd->p[i][j-1][k]*omhd->sy[i][j-1][k]/omhd->rho[i][j-1][k])+grid->difz[i][j][k]*(omhd->p[i][j][k+1]*omhd->sz[i][j][k+1]/omhd->rho[i][j][k+1]-omhd->p[i][j][k-1]*omhd->sz[i][j][k-1]/omhd->rho[i][j][k-1]))-(omhd->gamma[i][j][k]-1.)*omhd->p[i][j][k]*grid->dt*(grid->difx[i][j][k]*(omhd->sx[i+1][j][k]/omhd->rho[i+1][j][k]-omhd->sx[i-1][j][k]/omhd->rho[i-1][j][k])+grid->dify[i][j][k]*(omhd->sy[i][j+1][k]/omhd->rho[i][j+1][k]-omhd->sy[i][j-1][k]/omhd->rho[i][j-1][k])+grid->difz[i][j][k]*(omhd->sz[i][j][k+1]/omhd->rho[i][j][k+1]-omhd->sz[i][j][k-1]/omhd->rho[i][j][k-1]))+grid->dt*(omhd->gamma[i][j][k]-1.)*qed[i][j][k];
				mhd->pi[i][j][k]=omhd->pi[i][j][k]-grid->dt*(grid->difx[i][j][k]*(omhd->pi[i+1][j][k]*omhd->six[i+1][j][k]/omhd->rhoi[i+1][j][k]-omhd->pi[i-1][j][k]*omhd->six[i-1][j][k]/omhd->rhoi[i-1][j][k])+grid->dify[i][j][k]*(omhd->pi[i][j+1][k]*omhd->siy[i][j+1][k]/omhd->rhoi[i][j+1][k]-omhd->pi[i][j-1][k]*omhd->siy[i][j-1][k]/omhd->rhoi[i][j-1][k])+grid->difz[i][j][k]*(omhd->pi[i][j][k+1]*omhd->siz[i][j][k+1]/omhd->rhoi[i][j][k+1]-omhd->pi[i][j][k-1]*omhd->siz[i][j][k-1]/omhd->rhoi[i][j][k-1]))-(omhd->gammai[i][j][k]-1.)*omhd->pi[i][j][k]*grid->dt*(grid->difx[i][j][k]*(omhd->six[i+1][j][k]/omhd->rhoi[i+1][j][k]-omhd->six[i-1][j][k]/omhd->rhoi[i-1][j][k])+grid->dify[i][j][k]*(omhd->siy[i][j+1][k]/omhd->rhoi[i][j+1][k]-omhd->siy[i][j-1][k]/omhd->rhoi[i][j-1][k])+grid->difz[i][j][k]*(omhd->siz[i][j][k+1]/omhd->rhoi[i][j][k+1]-omhd->siz[i][j][k-1]/omhd->rhoi[i][j][k-1]))+grid->dt*(omhd->gammai[i][j][k]-1.)*qei[i][j][k];
				mhd->pe[i][j][k]=omhd->pe[i][j][k]-grid->dt*(grid->difx[i][j][k]*(omhd->pe[i+1][j][k]*omhd->sex[i+1][j][k]/omhd->rhoe[i+1][j][k]-omhd->pe[i-1][j][k]*omhd->sex[i-1][j][k]/omhd->rhoe[i-1][j][k])+grid->dify[i][j][k]*(omhd->pe[i][j+1][k]*omhd->sey[i][j+1][k]/omhd->rhoe[i][j+1][k]-omhd->pe[i][j-1][k]*omhd->sey[i][j-1][k]/omhd->rhoe[i][j-1][k])+grid->difz[i][j][k]*(omhd->pe[i][j][k+1]*omhd->sez[i][j][k+1]/omhd->rhoe[i][j][k+1]-omhd->pe[i][j][k-1]*omhd->sez[i][j][k-1]/omhd->rhoe[i][j][k-1]))-(omhd->gammae[i][j][k]-1.)*omhd->pe[i][j][k]*grid->dt*(grid->difx[i][j][k]*(omhd->sex[i+1][j][k]/omhd->rhoe[i+1][j][k]-omhd->sex[i-1][j][k]/omhd->rhoe[i-1][j][k])+grid->dify[i][j][k]*(omhd->sey[i][j+1][k]/omhd->rhoe[i][j+1][k]-omhd->sey[i][j-1][k]/omhd->rhoe[i][j-1][k])+grid->difz[i][j][k]*(omhd->sez[i][j][k+1]/omhd->rhoe[i][j][k+1]-omhd->sez[i][j][k-1]/omhd->rhoe[i][j][k-1]))+grid->dt*(omhd->gammae[i][j][k]-1.)*qee[i][j][k];
				mhd->pn[i][j][k]=omhd->pn[i][j][k]-grid->dt*(grid->difx[i][j][k]*(omhd->pn[i+1][j][k]*omhd->snx[i+1][j][k]/omhd->rhon[i+1][j][k]-omhd->pn[i-1][j][k]*omhd->snx[i-1][j][k]/omhd->rhon[i-1][j][k])+grid->dify[i][j][k]*(omhd->pn[i][j+1][k]*omhd->sny[i][j+1][k]/omhd->rhon[i][j+1][k]-omhd->pn[i][j-1][k]*omhd->sny[i][j-1][k]/omhd->rhon[i][j-1][k])+grid->difz[i][j][k]*(omhd->pn[i][j][k+1]*omhd->snz[i][j][k+1]/omhd->rhon[i][j][k+1]-omhd->pn[i][j][k-1]*omhd->snz[i][j][k-1]/omhd->rhon[i][j][k-1]))-(omhd->gamman[i][j][k]-1.)*omhd->pn[i][j][k]*grid->dt*(grid->difx[i][j][k]*(omhd->snx[i+1][j][k]/omhd->rhon[i+1][j][k]-omhd->snx[i-1][j][k]/omhd->rhon[i-1][j][k])+grid->dify[i][j][k]*(omhd->sny[i][j+1][k]/omhd->rhon[i][j+1][k]-omhd->sny[i][j-1][k]/omhd->rhon[i][j-1][k])+grid->difz[i][j][k]*(omhd->snz[i][j][k+1]/omhd->rhon[i][j][k+1]-omhd->snz[i][j][k-1]/omhd->rhon[i][j][k-1]))+grid->dt*(omhd->gamman[i][j][k]-1.)*qen[i][j][k];
			}
		}
	}
}
//*****************************************************************************
/*
	Function:	intscheme_nmhdust
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		9/26/08
	Inputs:		MHD mhd, GRID grid,int n, double t
	Outputs:	none
	Purpose:	This function organizes the leapfrog
			integration of the system of equations.
*/
void intscheme_nmhdust(MHD *mhd, GRID *grid, int *n, double *t)
{
	short int	test;

//	Perform 1st timestep
	test=0;
	leap_nmhdust(mhd,grid,test);
	*t=*t+grid->dt;
	(*n)++;
	bcorg_nmhdust(mhd,grid,t);
//      Perform 2nd timestep	
	test=1;
	leap_nmhdust(mhd,grid,test);
	*t=*t+grid->dt;
	(*n)++;
	bcorg_nmhdust(mhd,grid,t);
//      Check Values	
	valcheck(mhd,grid,n);
//	Calc nudi nuie nude
	if (grid->dnudi) nudicalc(mhd,grid);
	if (grid->dnude) nudecalc(mhd,grid);
	if (grid->dnuie) nuiecalc(mhd,grid);
	if (grid->dnudn) nudncalc(mhd,grid);
	if (grid->dnuin) nuincalc(mhd,grid);
	if (grid->dnuen) nuencalc(mhd,grid);
//      Below Here we can add diagnostic routines
	if ((*n < grid->lastball) && grid->ballon)
	{
        	ball(mhd,grid,n,t);
	}
//      Here we make a purturbation after lastball
        if ((*n > grid->lastball) && grid->ballon && grid->perturb)
	{
		grid->ballon=0;			//Turn Ball off so this only gets called once
		perturb(mhd,grid,t);
	}
	energy(mhd,grid,*t);
        *t=(*n)*grid->dt;
	if ((*n%grid->movieout) == 0)
	{
		movie(mhd,grid,*t);
	}
}
//*****************************************************************************
/*
	Function:	leap_nmhdust
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		10/08/08
	Inputs:		MHD mhd, GRID grid,int gchk
	Outputs:	none
	Purpose:	This function preforms the leap-frog integration.
			In addition to the leapfrog routine it applies a smoothing
			technique after the 2nd integration step.
	Notes:		9/29/08  SAL
				Fixed rhoi 2nd timestep error
			10/08/08 SAL
				Fixed Pressure Equation
			11/12/08 SAL
				Fixed ion vixen terms
			09/23/10 SAL
				Electron charge added
			09/30/10 SAL
				Added derivative source terms
			10/06/10 SAL
				Converted induction equation to electron velocity terms
			10/31/10 SAL
				Implemented OpenMP commands
			11/30/10 SAL
				Implemented u in place of p
*/
void	leap_nmhdust(MHD *mhd,GRID *grid,short int gchk)
{
	/*-- V A R I A B L E S --*/
	// Indexes
	int i,j,k,tid,nx,ny,nz;

	// Scalar Helpers
	double spat,memi,memd,mdpmi,mdpme,mipme,mdpmn,mipmn,mepmn;
	double mdinv,meinv,miinv,mimime,memime,mee,einv,dtinv;
	double stime,etime;

	// Helper Arrays
        double rhoinv[grid->nx][grid->ny][grid->nz],rhoiinv[grid->nx][grid->ny][grid->nz],rhoeinv[grid->nx][grid->ny][grid->nz],rhoninv[grid->nx][grid->ny][grid->nz];	// Rho Inversion
	double gaminv[grid->nx][grid->ny][grid->nz],gamiinv[grid->nx][grid->ny][grid->nz],gameinv[grid->nx][grid->ny][grid->nz],gamninv[grid->nx][grid->ny][grid->nz];	// Gamma Inversion 1/(gamma-1)
	double vx[grid->nx][grid->ny][grid->nz],vy[grid->nx][grid->ny][grid->nz],vz[grid->nx][grid->ny][grid->nz];					// V_dust
	double vix[grid->nx][grid->ny][grid->nz],viy[grid->nx][grid->ny][grid->nz],viz[grid->nx][grid->ny][grid->nz];					// V_ion
	double vex[grid->nx][grid->ny][grid->nz],vey[grid->nx][grid->ny][grid->nz],vez[grid->nx][grid->ny][grid->nz];					// V_electron
	double vnx[grid->nx][grid->ny][grid->nz],vny[grid->nx][grid->ny][grid->nz],vnz[grid->nx][grid->ny][grid->nz];					// V_neutral
	double rhovvxx[grid->nx][grid->ny][grid->nz],rhovvxy[grid->nx][grid->ny][grid->nz],rhovvxz[grid->nx][grid->ny][grid->nz];			// Dust Kinetic Energy Tensor
	double rhovvyy[grid->nx][grid->ny][grid->nz],rhovvyz[grid->nx][grid->ny][grid->nz],rhovvzz[grid->nx][grid->ny][grid->nz];
	double rhoivvxx[grid->nx][grid->ny][grid->nz],rhoivvxy[grid->nx][grid->ny][grid->nz],rhoivvxz[grid->nx][grid->ny][grid->nz];			// Ion Kinetic Energy Tensor
	double rhoivvyy[grid->nx][grid->ny][grid->nz],rhoivvyz[grid->nx][grid->ny][grid->nz],rhoivvzz[grid->nx][grid->ny][grid->nz];
	double rhonvvxx[grid->nx][grid->ny][grid->nz],rhonvvxy[grid->nx][grid->ny][grid->nz],rhonvvxz[grid->nx][grid->ny][grid->nz];			// Neutral Kinetic Energy Tensor
	double rhonvvyy[grid->nx][grid->ny][grid->nz],rhonvvyz[grid->nx][grid->ny][grid->nz],rhonvvzz[grid->nx][grid->ny][grid->nz];
	double vxbx[grid->nx][grid->ny][grid->nz],vxby[grid->nx][grid->ny][grid->nz],vxbz[grid->nx][grid->ny][grid->nz];				// Dust VXB Term
	double ivixbx[grid->nx][grid->ny][grid->nz],ivixby[grid->nx][grid->ny][grid->nz],ivixbz[grid->nx][grid->ny][grid->nz];			// Ion VXB Term
	double vexbx[grid->nx][grid->ny][grid->nz],vexby[grid->nx][grid->ny][grid->nz],vexbz[grid->nx][grid->ny][grid->nz];				// Dust VeXB Term
	double ivexbx[grid->nx][grid->ny][grid->nz],ivexby[grid->nx][grid->ny][grid->nz],ivexbz[grid->nx][grid->ny][grid->nz];			// Ion VeXB Term
	double bvexbx[grid->nx][grid->ny][grid->nz],bvexby[grid->nx][grid->ny][grid->nz],bvexbz[grid->nx][grid->ny][grid->nz];			// Induction VeXB Term
	double bvxbx[grid->nx][grid->ny][grid->nz],bvxby[grid->nx][grid->ny][grid->nz],bvxbz[grid->nx][grid->ny][grid->nz];			// Induction VxB Term
	double bvixbx[grid->nx][grid->ny][grid->nz],bvixby[grid->nx][grid->ny][grid->nz],bvixbz[grid->nx][grid->ny][grid->nz];			// Induction VxB Term
	double merhoe[grid->nx][grid->ny][grid->nz];								// Induction grad(pe) coefficient helper
	double srho[grid->nx][grid->ny][grid->nz],srhoi[grid->nx][grid->ny][grid->nz],srhoe[grid->nx][grid->ny][grid->nz],srhon[grid->nx][grid->ny][grid->nz];		// Saved Density Values

	// Derivative Helpers
        double drho[grid->nx][grid->ny][grid->nz],drhoi[grid->nx][grid->ny][grid->nz],drhoe[grid->nx][grid->ny][grid->nz],drhon[grid->nx][grid->ny][grid->nz];		// Density Derivatives
        double dsx[grid->nx][grid->ny][grid->nz],dsy[grid->nx][grid->ny][grid->nz],dsz[grid->nx][grid->ny][grid->nz];					// Dust Momentum Derivatives
        double dsix[grid->nx][grid->ny][grid->nz],dsiy[grid->nx][grid->ny][grid->nz],dsiz[grid->nx][grid->ny][grid->nz];				// Ion Momentum Derivatives
        double dsnx[grid->nx][grid->ny][grid->nz],dsny[grid->nx][grid->ny][grid->nz],dsnz[grid->nx][grid->ny][grid->nz];				// Neutral Momentum Derivatives
        double dp[grid->nx][grid->ny][grid->nz],dpi[grid->nx][grid->ny][grid->nz],dpe[grid->nx][grid->ny][grid->nz],dpn[grid->nx][grid->ny][grid->nz];			// Pressure Derivatives
        double dbx[grid->nx][grid->ny][grid->nz],dby[grid->nx][grid->ny][grid->nz],dbz[grid->nx][grid->ny][grid->nz];					// Induction Derivatives
        double avgped[grid->nx][grid->ny][grid->nz];													// Spatial Average of momentum pe coefficient (dust)
        double avgpei[grid->nx][grid->ny][grid->nz];													// Spatial Average of momentum pe coefficient (ion)
        double avgdsx[grid->nx][grid->ny][grid->nz],avgdsy[grid->nx][grid->ny][grid->nz],avgdsz[grid->nx][grid->ny][grid->nz];				// Spatial Average of total pe term 
        double avgdsix[grid->nx][grid->ny][grid->nz],avgdsiy[grid->nx][grid->ny][grid->nz],avgdsiz[grid->nx][grid->ny][grid->nz];			// Spatial Average of total pe term
        double cdsx[grid->nx][grid->ny][grid->nz],cdsy[grid->nx][grid->ny][grid->nz],cdsz[grid->nx][grid->ny][grid->nz];				// Centered pe term
        double cdsix[grid->nx][grid->ny][grid->nz],cdsiy[grid->nx][grid->ny][grid->nz],cdsiz[grid->nx][grid->ny][grid->nz];				// Centered pe term
        double avgdpd[grid->nx][grid->ny][grid->nz];
        double avgdpi[grid->nx][grid->ny][grid->nz];
        double avgdpe[grid->nx][grid->ny][grid->nz];
        double avgdpn[grid->nx][grid->ny][grid->nz];
        double cdpd[grid->nx][grid->ny][grid->nz];
        double cdpi[grid->nx][grid->ny][grid->nz];
        double cdpe[grid->nx][grid->ny][grid->nz];
        double cdpn[grid->nx][grid->ny][grid->nz];

	// Source Term Helpers
	double qci[grid->nx][grid->ny][grid->nz],qcn[grid->nx][grid->ny][grid->nz];							// Density Q_C Terms
	double qsdx[grid->nx][grid->ny][grid->nz],qsdy[grid->nx][grid->ny][grid->nz],qsdz[grid->nx][grid->ny][grid->nz];				// Dust Momentum Q_S Term
	double qsix[grid->nx][grid->ny][grid->nz],qsiy[grid->nx][grid->ny][grid->nz],qsiz[grid->nx][grid->ny][grid->nz];				// Ion Momentum Q_S Term
	double qsex[grid->nx][grid->ny][grid->nz],qsey[grid->nx][grid->ny][grid->nz],qsez[grid->nx][grid->ny][grid->nz];				// Electron Momentum Q_S Term
	double qsnx[grid->nx][grid->ny][grid->nz],qsny[grid->nx][grid->ny][grid->nz],qsnz[grid->nx][grid->ny][grid->nz];				// Neutral Momentum Q_S Term
	double qed[grid->nx][grid->ny][grid->nz],qei[grid->nx][grid->ny][grid->nz],qee[grid->nx][grid->ny][grid->nz],qen[grid->nx][grid->ny][grid->nz];			// Energy Q_E Terms
	double avgqci[grid->nx][grid->ny][grid->nz],avgqcn[grid->nx][grid->ny][grid->nz];						// Averaged Continuity Source Terms
	double avgqsx[grid->nx][grid->ny][grid->nz],avgqsy[grid->nx][grid->ny][grid->nz],avgqsz[grid->nx][grid->ny][grid->nz];			// Averaged Dust Momentum Source Term
	double avgqsix[grid->nx][grid->ny][grid->nz],avgqsiy[grid->nx][grid->ny][grid->nz],avgqsiz[grid->nx][grid->ny][grid->nz];			// Averaged Ion Momentum Source Term
        double avgqsnx[grid->nx][grid->ny][grid->nz],avgqsny[grid->nx][grid->ny][grid->nz],avgqsnz[grid->nx][grid->ny][grid->nz];			// Averaged Neutral Momentum Source Term
	double avgqed[grid->nx][grid->ny][grid->nz],avgqei[grid->nx][grid->ny][grid->nz],avgqee[grid->nx][grid->ny][grid->nz],avgqen[grid->nx][grid->ny][grid->nz];	// Averaged Pressure Source Terms
	
	// Calc Some Helper Arrays
	dtinv=1/grid->dt;
	spat=1./6.;
	memd=mhd->me/mhd->md;
	memi=mhd->me/mhd->mi;
	mdpmi=1./(mhd->md+mhd->mi);
	mdpme=1./(mhd->md+mhd->me);
	mipme=1./(mhd->mi+mhd->me);
	mdpmn=1./(mhd->md+mhd->mn);
	mipmn=1./(mhd->mi+mhd->mn);
	mepmn=1./(mhd->me+mhd->mn);
        mdinv=1./mhd->md;
	meinv=1./mhd->me;
	miinv=1./mhd->mi;
	einv=1./mhd->e;
	mimime=mhd->mi*mipme;
	memime=mhd->me*mipme;
	mee=mhd->me*einv;
	//If gchk=0 then we work on even grid points if gchk=1 then we work on odd grid points
	//Calc some helper arrays (on all grids, can probably get broken up a bit)

	#pragma omp parallel default(shared) private(i,j,k,tid,stime) 
	{
//		tid=omp_get_thread_num();
//		printf("-----  Thread %d Checking in!\n",tid);
//		#pragma omp barrier
//		if (tid == 0)
//		{
//			stime=omp_get_wtime();
//			printf("----- Calculating inversions");
//		}
//		#pragma omp barrier
		#pragma omp for
		for(i=0;i<grid->nx;i++)
		{
			for(j=0;j<grid->ny;j++)
			{
				for(k=0;k<grid->nz;k++)
				{
					//      rho's
					rhoinv[i][j][k]=1./mhd->rho[i][j][k];
					rhoiinv[i][j][k]=1./mhd->rhoi[i][j][k];
					rhoeinv[i][j][k]=1./mhd->rhoe[i][j][k];
					rhoninv[i][j][k]=1./mhd->rhon[i][j][k];
					//	1./(gamma-1.)
					gaminv[i][j][k]=1./(mhd->gamma[i][j][k]-1.);
					gamiinv[i][j][k]=1./(mhd->gammai[i][j][k]-1.);
					//gameinv[i][j][k]=1./(mhd->gammae[i][j][k]-1.);
					gameinv[i][j][k]=1.;
					gamninv[i][j][k]=1./(mhd->gamman[i][j][k]-1.);
					
				}
			}
		}
//		if (tid == 0)
//		{
//			printf(".....%04.2f [s] -----\n",(omp_get_wtime()-stime)/TIMESCALE);
//			stime=omp_get_wtime();
//			printf("----- Calculating leap helpers");
//		}
		#pragma omp for
		for(i=0;i<grid->nx;i++)
		{
			for(j=0;j<grid->ny;j++)
			{
				for(k=0;k<grid->nz;k++)
				{
					//      v's
					vx[i][j][k]=mhd->sx[i][j][k]*rhoinv[i][j][k];
					vy[i][j][k]=mhd->sy[i][j][k]*rhoinv[i][j][k];
					vz[i][j][k]=mhd->sz[i][j][k]*rhoinv[i][j][k];
					vix[i][j][k]=mhd->six[i][j][k]*rhoiinv[i][j][k];
					viy[i][j][k]=mhd->siy[i][j][k]*rhoiinv[i][j][k];
					viz[i][j][k]=mhd->siz[i][j][k]*rhoiinv[i][j][k];
					vex[i][j][k]=mhd->sex[i][j][k]*rhoeinv[i][j][k];
					vey[i][j][k]=mhd->sey[i][j][k]*rhoeinv[i][j][k];
					vez[i][j][k]=mhd->sez[i][j][k]*rhoeinv[i][j][k];
					vnx[i][j][k]=mhd->snx[i][j][k]*rhoninv[i][j][k];
					vny[i][j][k]=mhd->sny[i][j][k]*rhoninv[i][j][k];
					vnz[i][j][k]=mhd->snz[i][j][k]*rhoninv[i][j][k];
				}
			}
		}
//		if (tid == 0)
//		{
//			printf("..........%04.2f [s] -----\n",(omp_get_wtime()-stime)/TIMESCALE);
//			stime=omp_get_wtime();
//			printf("----- Calculating source term helpers");
//		}
//		#pragma omp barrier
		#pragma omp for
		for(i=0;i<grid->nx;i++)
		{
			for(j=0;j<grid->ny;j++)
			{
				for(k=0;k<grid->nz;k++)
				{
					//	rho*v*v
					rhovvxx[i][j][k]=mhd->sx[i][j][k]*mhd->sx[i][j][k]*rhoinv[i][j][k];
					rhovvxy[i][j][k]=mhd->sx[i][j][k]*mhd->sy[i][j][k]*rhoinv[i][j][k];
					rhovvxz[i][j][k]=mhd->sx[i][j][k]*mhd->sz[i][j][k]*rhoinv[i][j][k];
					rhovvyy[i][j][k]=mhd->sy[i][j][k]*mhd->sy[i][j][k]*rhoinv[i][j][k];
					rhovvyz[i][j][k]=mhd->sy[i][j][k]*mhd->sz[i][j][k]*rhoinv[i][j][k];
					rhovvzz[i][j][k]=mhd->sz[i][j][k]*mhd->sz[i][j][k]*rhoinv[i][j][k];
					//	vxb term
					vxbx[i][j][k]=mhd->zd[i][j][k]*mhd->e*mdinv*(mhd->sy[i][j][k]*mhd->bz[i][j][k]-mhd->sz[i][j][k]*mhd->by[i][j][k]);
					vxby[i][j][k]=mhd->zd[i][j][k]*mhd->e*mdinv*(mhd->sz[i][j][k]*mhd->bx[i][j][k]-mhd->sx[i][j][k]*mhd->bz[i][j][k]);
					vxbz[i][j][k]=mhd->zd[i][j][k]*mhd->e*mdinv*(mhd->sx[i][j][k]*mhd->by[i][j][k]-mhd->sy[i][j][k]*mhd->bx[i][j][k]);
					//	vexb term
					vexbx[i][j][k]=mhd->zd[i][j][k]*mhd->e*mdinv*mhd->rho[i][j][k]*(vey[i][j][k]*mhd->bz[i][j][k]-vez[i][j][k]*mhd->by[i][j][k]);
					vexby[i][j][k]=mhd->zd[i][j][k]*mhd->e*mdinv*mhd->rho[i][j][k]*(vez[i][j][k]*mhd->bx[i][j][k]-vex[i][j][k]*mhd->bz[i][j][k]);
					vexbz[i][j][k]=mhd->zd[i][j][k]*mhd->e*mdinv*mhd->rho[i][j][k]*(vex[i][j][k]*mhd->by[i][j][k]-vey[i][j][k]*mhd->bx[i][j][k]);
					//      rhoi*vi*vi
					rhoivvxx[i][j][k]=mhd->six[i][j][k]*mhd->six[i][j][k]*rhoiinv[i][j][k];
					rhoivvxy[i][j][k]=mhd->six[i][j][k]*mhd->siy[i][j][k]*rhoiinv[i][j][k];
					rhoivvxz[i][j][k]=mhd->six[i][j][k]*mhd->siz[i][j][k]*rhoiinv[i][j][k];
					rhoivvyy[i][j][k]=mhd->siy[i][j][k]*mhd->siy[i][j][k]*rhoiinv[i][j][k];
					rhoivvyz[i][j][k]=mhd->siy[i][j][k]*mhd->siz[i][j][k]*rhoiinv[i][j][k];
					rhoivvzz[i][j][k]=mhd->siz[i][j][k]*mhd->siz[i][j][k]*rhoiinv[i][j][k];
					//	ivixb term
					ivixbx[i][j][k]=mhd->zi[i][j][k]*mhd->e*miinv*(mhd->siy[i][j][k]*mhd->bz[i][j][k]-mhd->siz[i][j][k]*mhd->by[i][j][k]);
					ivixby[i][j][k]=mhd->zi[i][j][k]*mhd->e*miinv*(mhd->siz[i][j][k]*mhd->bx[i][j][k]-mhd->six[i][j][k]*mhd->bz[i][j][k]);
					ivixbz[i][j][k]=mhd->zi[i][j][k]*mhd->e*miinv*(mhd->six[i][j][k]*mhd->by[i][j][k]-mhd->siy[i][j][k]*mhd->bx[i][j][k]);
					//	ivexb term
					ivexbx[i][j][k]=mhd->zi[i][j][k]*mhd->e*miinv*mhd->rhoi[i][j][k]*(vey[i][j][k]*mhd->bz[i][j][k]-vez[i][j][k]*mhd->by[i][j][k]);
					ivexby[i][j][k]=mhd->zi[i][j][k]*mhd->e*miinv*mhd->rhoi[i][j][k]*(vez[i][j][k]*mhd->bx[i][j][k]-vex[i][j][k]*mhd->bz[i][j][k]);
					ivexbz[i][j][k]=mhd->zi[i][j][k]*mhd->e*miinv*mhd->rhoi[i][j][k]*(vex[i][j][k]*mhd->by[i][j][k]-vey[i][j][k]*mhd->bx[i][j][k]);
					//      rhon*vn*vn
					rhonvvxx[i][j][k]=mhd->rhon[i][j][k]*vnx[i][j][k]*vnx[i][j][k];
					rhonvvxy[i][j][k]=mhd->rhon[i][j][k]*vnx[i][j][k]*vny[i][j][k];
					rhonvvxz[i][j][k]=mhd->rhon[i][j][k]*vnx[i][j][k]*vnz[i][j][k];
					rhonvvyy[i][j][k]=mhd->rhon[i][j][k]*vny[i][j][k]*vny[i][j][k];
					rhonvvyz[i][j][k]=mhd->rhon[i][j][k]*vny[i][j][k]*vnz[i][j][k];
					rhonvvzz[i][j][k]=mhd->rhon[i][j][k]*vnz[i][j][k]*vnz[i][j][k];
					//     Induction Helpers
					merhoe[i][j][k]=mhd->me*rhoeinv[i][j][k]*rhoeinv[i][j][k];
					bvexbx[i][j][k]=vey[i][j][k]*mhd->bz[i][j][k]-vez[i][j][k]*mhd->by[i][j][k];
					bvexby[i][j][k]=vez[i][j][k]*mhd->bx[i][j][k]-vex[i][j][k]*mhd->bz[i][j][k];
					bvexbz[i][j][k]=vex[i][j][k]*mhd->by[i][j][k]-vey[i][j][k]*mhd->bx[i][j][k];
//					bvxbx[i][j][k]=mhd->zd[i][j][k]*mhd->me*rhoeinv[i][j][k]*(mhd->sy[i][j][k]*mhd->bz[i][j][k]-mhd->sz[i][j][k]*mhd->by[i][j][k]);
//					bvxbx[i][j][k]=mhd->zd[i][j][k]*mhd->me*rhoeinv[i][j][k]*(mhd->sz[i][j][k]*mhd->bx[i][j][k]-mhd->sx[i][j][k]*mhd->bz[i][j][k]);
//					bvxbx[i][j][k]=mhd->zd[i][j][k]*mhd->me*rhoeinv[i][j][k]*(mhd->sx[i][j][k]*mhd->by[i][j][k]-mhd->sy[i][j][k]*mhd->bx[i][j][k]);
//					bvixbx[i][j][k]=mhd->zi[i][j][k]*mhd->me*rhoeinv[i][j][k]*(mhd->siy[i][j][k]*mhd->bz[i][j][k]-mhd->siz[i][j][k]*mhd->by[i][j][k]);
//					bvixbx[i][j][k]=mhd->zi[i][j][k]*mhd->me*rhoeinv[i][j][k]*(mhd->siz[i][j][k]*mhd->bx[i][j][k]-mhd->six[i][j][k]*mhd->bz[i][j][k]);
//					bvixbx[i][j][k]=mhd->zi[i][j][k]*mhd->me*rhoeinv[i][j][k]*(mhd->six[i][j][k]*mhd->by[i][j][k]-mhd->siy[i][j][k]*mhd->bx[i][j][k]);
					
					//     Collisional & Ionization Source Terms
					//		qck:  Continuity
					//		qsk:  Momentum
					//		qek:  Engery
					qci[i][j][k]=mimime*(mhd->ioniz*mhd->rhon[i][j][k]-mhd->recom*mhd->rhoi[i][j][k]*mhd->rhoe[i][j][k]*meinv);
					qcn[i][j][k]=-1.*mhd->ioniz*mhd->rhon[i][j][k]+mhd->recom*mhd->rhoi[i][j][k]*mhd->rhoe[i][j][k]*meinv;
					qsdx[i][j][k]=-1.*mhd->rhoi[i][j][k]*mhd->nuid[i][j][k]*mdpmi*(vx[i][j][k]-vix[i][j][k])
							 -mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*(vx[i][j][k]-vex[i][j][k])
							 -mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*(vx[i][j][k]-vnx[i][j][k]);
					qsdy[i][j][k]=-1.*mhd->rhoi[i][j][k]*mhd->nuid[i][j][k]*mdpmi*(vy[i][j][k]-viy[i][j][k])
							 -mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*(vy[i][j][k]-vey[i][j][k])
							 -mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*(vy[i][j][k]-vny[i][j][k]);
					qsdz[i][j][k]=-1.*mhd->rhoi[i][j][k]*mhd->nuid[i][j][k]*mdpmi*(vz[i][j][k]-viz[i][j][k])
							 -mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*(vz[i][j][k]-vez[i][j][k])
							 -mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*(vz[i][j][k]-vnz[i][j][k]);
					qsix[i][j][k]=-1.*mhd->rhoi[i][j][k]*mhd->nuid[i][j][k]*mdpmi*(vix[i][j][k]-vx[i][j][k])
							 -mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*mipme*(vix[i][j][k]-vex[i][j][k])
							 -mhd->rhoi[i][j][k]*mhd->nuin[i][j][k]*mipmn*(vix[i][j][k]-vnx[i][j][k])
							 +mhd->mi*mipme*mhd->ioniz*mhd->snx[i][j][k]
							 -mimime*mhd->recom*mhd->six[i][j][k]*mhd->rhoe[i][j][k]*meinv;
					qsiy[i][j][k]=-1.*mhd->rhoi[i][j][k]*mhd->nuid[i][j][k]*mdpmi*(viy[i][j][k]-vy[i][j][k])
							 -mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*mipme*(viy[i][j][k]-vey[i][j][k])
							 -mhd->rhoi[i][j][k]*mhd->nuin[i][j][k]*mipmn*(viy[i][j][k]-vny[i][j][k])
							 +mhd->mi*mipme*mhd->ioniz*mhd->sny[i][j][k]
							 -mimime*mhd->recom*mhd->siy[i][j][k]*mhd->rhoe[i][j][k]*meinv;
					qsiz[i][j][k]=-1.*mhd->rhoi[i][j][k]*mhd->nuid[i][j][k]*mdpmi*(viz[i][j][k]-vz[i][j][k])
							 -mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*mipme*(viz[i][j][k]-vez[i][j][k])
							 -mhd->rhoi[i][j][k]*mhd->nuin[i][j][k]*mipmn*(viz[i][j][k]-vnz[i][j][k])
							 +mhd->mi*mipme*mhd->ioniz*mhd->snz[i][j][k]
							 -mimime*mhd->recom*mhd->siz[i][j][k]*mhd->rhoe[i][j][k]*meinv;
					qsex[i][j][k]=-1.*mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*(vex[i][j][k]-vx[i][j][k])
							 -mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*mipme*(vex[i][j][k]-vix[i][j][k])
							 -mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*(vex[i][j][k]-vnx[i][j][k])
							 +mhd->me*mipme*mhd->ioniz*mhd->snx[i][j][k]
							 -mipme*mhd->recom*mhd->rhoi[i][j][k]*mhd->sex[i][j][k];
					qsey[i][j][k]=-1.*mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*(vey[i][j][k]-vy[i][j][k])
							 -mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*mipme*(vey[i][j][k]-viy[i][j][k])
							 -mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*(vey[i][j][k]-vny[i][j][k])
							 +mhd->me*mipme*mhd->ioniz*mhd->sny[i][j][k]
							 -mipme*mhd->recom*mhd->rhoi[i][j][k]*mhd->sey[i][j][k];
					qsez[i][j][k]=-1.*mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*(vez[i][j][k]-vz[i][j][k])
							 -mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*mipme*(vez[i][j][k]-viz[i][j][k])
							 -mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*(vez[i][j][k]-vnz[i][j][k])
							 +mhd->me*mipme*mhd->ioniz*mhd->snz[i][j][k]
							 -mipme*mhd->recom*mhd->rhoi[i][j][k]*mhd->sez[i][j][k];
					qsnx[i][j][k]=-1.*mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*(vnx[i][j][k]-vx[i][j][k])
							 -mhd->rhoi[i][j][k]*mhd->nuin[i][j][k]*mipmn*(vnx[i][j][k]-vix[i][j][k])
							 -mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*(vnx[i][j][k]-vex[i][j][k])
							 -mhd->ioniz*mhd->snx[i][j][k]
							 +mhd->recom*mipme*(mhd->six[i][j][k]*mhd->rhoe[i][j][k]/memi+mhd->sex[i][j][k]*mhd->rhoi[i][j][k]);
					qsny[i][j][k]=-1.*mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*(vny[i][j][k]-vy[i][j][k])
							 -mhd->rhoi[i][j][k]*mhd->nuin[i][j][k]*mipmn*(vny[i][j][k]-viy[i][j][k])
							 -mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*(vny[i][j][k]-vey[i][j][k])
							 -mhd->ioniz*mhd->sny[i][j][k]
							 +mhd->recom*mipme*(mhd->siy[i][j][k]*mhd->rhoe[i][j][k]/memi+mhd->sey[i][j][k]*mhd->rhoi[i][j][k]);
					qsnz[i][j][k]=-1.*mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*(vnz[i][j][k]-vz[i][j][k])
							 -mhd->rhoi[i][j][k]*mhd->nuin[i][j][k]*mipmn*(vnz[i][j][k]-viz[i][j][k])
							 -mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*(vnz[i][j][k]-vez[i][j][k])
							 -mhd->ioniz*mhd->snz[i][j][k]
							 +mhd->recom*mipme*(mhd->siz[i][j][k]*mhd->rhoe[i][j][k]/memi+mhd->sez[i][j][k]*mhd->rhoi[i][j][k]);
					qed[i][j][k]=mhd->rhoi[i][j][k]*mhd->nuid[i][j][k]*mdpmi*( (vix[i][j][k]-vx[i][j][k])*(vix[i][j][k]-vx[i][j][k])
												  +(viy[i][j][k]-vy[i][j][k])*(viy[i][j][k]-vy[i][j][k])
												  +(viz[i][j][k]-vz[i][j][k])*(viz[i][j][k]-vz[i][j][k]))
						    +mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*( (vex[i][j][k]-vx[i][j][k])*(vex[i][j][k]-vx[i][j][k])
												 +(vey[i][j][k]-vy[i][j][k])*(vey[i][j][k]-vy[i][j][k])
												 +(vez[i][j][k]-vz[i][j][k])*(vez[i][j][k]-vz[i][j][k]))
						    +mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*( (vnx[i][j][k]-vx[i][j][k])*(vnx[i][j][k]-vx[i][j][k])
												 +(vny[i][j][k]-vy[i][j][k])*(vny[i][j][k]-vy[i][j][k])
												 +(vnz[i][j][k]-vz[i][j][k])*(vnz[i][j][k]-vz[i][j][k]))
						    -2.*mhd->rhoi[i][j][k]*mhd->nuid[i][j][k]*mdpmi*(mhd->p[i][j][k]*gaminv[i][j][k]*rhoinv[i][j][k]-mhd->mi*mhd->pi[i][j][k]*gamiinv[i][j][k]*rhoiinv[i][j][k])
						    -2.*mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*(mhd->p[i][j][k]*gaminv[i][j][k]*rhoinv[i][j][k]-mhd->me*mhd->pe[i][j][k]*gameinv[i][j][k]*rhoeinv[i][j][k])
						    -2.*mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*(mhd->p[i][j][k]*gaminv[i][j][k]*rhoinv[i][j][k]-mhd->mn*mhd->pn[i][j][k]*gamninv[i][j][k]*rhoninv[i][j][k]);
					qei[i][j][k]=mhd->rhoi[i][j][k]*( mhd->nuid[i][j][k]*mdpmi*( (vx[i][j][k]-vix[i][j][k])*(vx[i][j][k]-vix[i][j][k])
												   +(vy[i][j][k]-viy[i][j][k])*(vy[i][j][k]-viy[i][j][k])
												   +(vz[i][j][k]-viz[i][j][k])*(vz[i][j][k]-viz[i][j][k]))
									 +mhd->nuie[i][j][k]*mipme*( (vex[i][j][k]-vix[i][j][k])*(vex[i][j][k]-vix[i][j][k])
												    +(vey[i][j][k]-viy[i][j][k])*(vey[i][j][k]-viy[i][j][k])
												    +(vez[i][j][k]-viz[i][j][k])*(vez[i][j][k]-viz[i][j][k]))
									 +mhd->nuin[i][j][k]*mipmn*( (vnx[i][j][k]-vix[i][j][k])*(vnx[i][j][k]-vix[i][j][k])
												    +(vny[i][j][k]-viy[i][j][k])*(vny[i][j][k]-viy[i][j][k])
												    +(vnz[i][j][k]-viz[i][j][k])*(vnz[i][j][k]-viz[i][j][k])))
						    -2.*mhd->rhoi[i][j][k]*mhd->nuid[i][j][k]*mdpmi*(mhd->mi*mhd->pi[i][j][k]*gamiinv[i][j][k]*rhoiinv[i][j][k]-mhd->p[i][j][k]*gaminv[i][j][k]*rhoinv[i][j][k])
						    -2.*mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*mipme*(mhd->mi*mhd->pi[i][j][k]*gamiinv[i][j][k]*rhoiinv[i][j][k]-mhd->me*mhd->pe[i][j][k]*gameinv[i][j][k]*rhoeinv[i][j][k])
						    -2.*mhd->rhoi[i][j][k]*mhd->nuin[i][j][k]*mipmn*(mhd->mi*mhd->pi[i][j][k]*gamiinv[i][j][k]*rhoiinv[i][j][k]-mhd->mn*mhd->pn[i][j][k]*gamninv[i][j][k]*rhoninv[i][j][k])
						    +mhd->gamman[i][j][k]*mipme*mhd->pn[i][j][k]*mhd->ioniz*mhd->mn*gamninv[i][j][k]
						    -mhd->gammai[i][j][k]*mhd->pi[i][j][k]*mhd->recom*mhd->rhoe[i][j][k]*meinv*gamiinv[i][j][k]
						    +0.25*mhd->mi*mipme*( mhd->ioniz*mhd->rhon[i][j][k]
									 +mhd->recom*mhd->rhoi[i][j][k]*mhd->rhoe[i][j][k]*meinv)*( (vnx[i][j][k]-vix[i][j][k])*(vnx[i][j][k]-vix[i][j][k])
																   +(vny[i][j][k]-viy[i][j][k])*(vny[i][j][k]-viy[i][j][k])
																   +(vnz[i][j][k]-viz[i][j][k])*(vnz[i][j][k]-viz[i][j][k]));
					qee[i][j][k]= mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*( (vx[i][j][k]-vex[i][j][k])*(vx[i][j][k]-vex[i][j][k])
												  +(vy[i][j][k]-vey[i][j][k])*(vy[i][j][k]-vey[i][j][k])
												  +(vz[i][j][k]-vez[i][j][k])*(vz[i][j][k]-vez[i][j][k]))
						    +mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*mipme*( (vix[i][j][k]-vex[i][j][k])*(vix[i][j][k]-vex[i][j][k])
												  +(viy[i][j][k]-vey[i][j][k])*(viy[i][j][k]-vey[i][j][k])
												  +(viz[i][j][k]-vez[i][j][k])*(viz[i][j][k]-vez[i][j][k]))
						    +mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*( (vnx[i][j][k]-vex[i][j][k])*(vnx[i][j][k]-vex[i][j][k])
												  +(vny[i][j][k]-vey[i][j][k])*(vny[i][j][k]-vey[i][j][k])
												  +(vnz[i][j][k]-vez[i][j][k])*(vnz[i][j][k]-vez[i][j][k]))
						    -2.*mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*(mhd->me*mhd->pe[i][j][k]*gameinv[i][j][k]*rhoeinv[i][j][k]-mhd->p[i][j][k]*gaminv[i][j][k]*rhoinv[i][j][k])
						    -2.*mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*mipme*(mhd->me*mhd->pe[i][j][k]*gameinv[i][j][k]*rhoeinv[i][j][k]-mhd->mi*mhd->pi[i][j][k]*gamiinv[i][j][k]*rhoiinv[i][j][k])
						    -2.*mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*(mhd->me*mhd->pe[i][j][k]*gameinv[i][j][k]*rhoeinv[i][j][k]-mhd->mn*mhd->pn[i][j][k]*gamninv[i][j][k]*rhoninv[i][j][k])
						    +mhd->gamman[i][j][k]*mipme*mhd->pn[i][j][k]*mhd->ioniz*memi*mhd->mn*gamninv[i][j][k]
						    -mhd->gammae[i][j][k]*mhd->pe[i][j][k]*mhd->recom*mhd->rhoi[i][j][k]*miinv*gameinv[i][j][k]
						    +0.25*mipme*( mhd->ioniz*mhd->rhon[i][j][k]*mhd->me
								 +mhd->recom*mhd->rhoi[i][j][k]*mhd->rhoe[i][j][k])*( (vnx[i][j][k]-vex[i][j][k])*(vnx[i][j][k]-vex[i][j][k])
														     +(vny[i][j][k]-vey[i][j][k])*(vny[i][j][k]-vey[i][j][k])
														     +(vnz[i][j][k]-vez[i][j][k])*(vnz[i][j][k]-vez[i][j][k]));
					qen[i][j][k]= mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*(  (vnx[i][j][k]-vx[i][j][k])*(vnx[i][j][k]-vx[i][j][k])
												   +(vny[i][j][k]-vy[i][j][k])*(vny[i][j][k]-vy[i][j][k])
												   +(vnz[i][j][k]-vz[i][j][k])*(vnz[i][j][k]-vz[i][j][k]))
						     +mhd->rhoi[i][j][k]*mhd->nuin[i][j][k]*mipmn*( (vnx[i][j][k]-vix[i][j][k])*(vnx[i][j][k]-vix[i][j][k])
												   +(vny[i][j][k]-viy[i][j][k])*(vny[i][j][k]-viy[i][j][k])
												   +(vnz[i][j][k]-viz[i][j][k])*(vnz[i][j][k]-viz[i][j][k]))
						     +mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*( (vnx[i][j][k]-vex[i][j][k])*(vnx[i][j][k]-vex[i][j][k])
												   +(vny[i][j][k]-vey[i][j][k])*(vny[i][j][k]-vey[i][j][k])
												   +(vnz[i][j][k]-vez[i][j][k])*(vnz[i][j][k]-vez[i][j][k]))
						     -2.*mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*(mhd->mn*mhd->pn[i][j][k]*rhoninv[i][j][k]*gamninv[i][j][k]-mhd->p[i][j][k]*rhoinv[i][j][k]*gaminv[i][j][k])
						     -2.*mhd->rhoi[i][j][k]*mhd->nuin[i][j][k]*mipmn*(mhd->mn*mhd->pn[i][j][k]*rhoninv[i][j][k]*gamninv[i][j][k]-mhd->mi*mhd->pi[i][j][k]*rhoiinv[i][j][k]*gamiinv[i][j][k])
						     -2.*mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*(mhd->mn*mhd->pn[i][j][k]*rhoninv[i][j][k]*gamninv[i][j][k]-mhd->me*mhd->pe[i][j][k]*rhoeinv[i][j][k]*gameinv[i][j][k])
						     -mhd->gamman[i][j][k]*mhd->pn[i][j][k]*mhd->ioniz*gamninv[i][j][k]
						     +( mhd->gammae[i][j][k]*mhd->pe[i][j][k]*rhoeinv[i][j][k]*mhd->me*gameinv[i][j][k]
						      +mhd->gammai[i][j][k]*mhd->mi*mhd->pi[i][j][k]*rhoiinv[i][j][k]*gamiinv[i][j][k])*mhd->recom*mhd->rhoi[i][j][k]*mhd->rhoe[i][j][k]*miinv*meinv
						     +0.25*mhd->mi*mipme*( mhd->ioniz*mhd->rhon[i][j][k]
									  +mhd->recom*mhd->rhoi[i][j][k]*mhd->rhoe[i][j][k]*meinv)*( (vnx[i][j][k]-vix[i][j][k])*(vnx[i][j][k]-vix[i][j][k])
																    +(vny[i][j][k]-viy[i][j][k])*(vny[i][j][k]-viy[i][j][k])
																    +(vnz[i][j][k]-viz[i][j][k])*(vnz[i][j][k]-viz[i][j][k]))
						     +0.25*mipme*( mhd->ioniz*mhd->rhon[i][j][k]*mhd->me
								  +mhd->recom*mhd->rhoi[i][j][k]*mhd->rhoe[i][j][k])*( (vnx[i][j][k]-vex[i][j][k])*(vnx[i][j][k]-vex[i][j][k])
														      +(vny[i][j][k]-vey[i][j][k])*(vny[i][j][k]-vey[i][j][k])
														      +(vnz[i][j][k]-vez[i][j][k])*(vnz[i][j][k]-vez[i][j][k]));
				}
			}
		}
//		if (tid == 0)
//		{
//			printf("...%04.2f [s] -----\n",(omp_get_wtime()-stime)/TIMESCALE);
//			stime=omp_get_wtime();
//			printf("----- Calculating average helpers");
//		}
		//Calc Averaged Pe and Pd terms
		#pragma omp for
		for(i=1;i<grid->nxm1;i++)
		{
			for(j=1;j<grid->nym1;j++)
			{
				for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
				{
					avgped[i][j][k]=spat * memd * (  mhd->zd[i+1][j][k] * mhd->rho[i+1][j][k] * rhoeinv[i+1][j][k]
						                       + mhd->zd[i-1][j][k] * mhd->rho[i-1][j][k] * rhoeinv[i-1][j][k]
						                       + mhd->zd[i][j+1][k] * mhd->rho[i][j+1][k] * rhoeinv[i][j+1][k]
						                       + mhd->zd[i][j-1][k] * mhd->rho[i][j-1][k] * rhoeinv[i][j-1][k]
						                       + mhd->zd[i][j][k+1] * mhd->rho[i][j][k+1] * rhoeinv[i][j][k+1]
						                       + mhd->zd[i][j][k-1] * mhd->rho[i][j][k-1] * rhoeinv[i][j][k-1]);
					avgpei[i][j][k]=spat * memi * (  mhd->zi[i+1][j][k] * mhd->rhoi[i+1][j][k] * rhoeinv[i+1][j][k]
						                       + mhd->zi[i-1][j][k] * mhd->rhoi[i-1][j][k] * rhoeinv[i-1][j][k]
						                       + mhd->zi[i][j+1][k] * mhd->rhoi[i][j+1][k] * rhoeinv[i][j+1][k]
						                       + mhd->zi[i][j-1][k] * mhd->rhoi[i][j-1][k] * rhoeinv[i][j-1][k]
						                       + mhd->zi[i][j][k+1] * mhd->rhoi[i][j][k+1] * rhoeinv[i][j][k+1]
						                       + mhd->zi[i][j][k-1] * mhd->rhoi[i][j][k-1] * rhoeinv[i][j][k-1]);
				}
			}
		}
		//Calc Averaged Source Terms
//		#pragma omp barrier
		#pragma omp for
		for(i=1;i<grid->nxm1;i++)
		{
			for(j=1;j<grid->nym1;j++)
			{
				for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
				{
					// Continuity
					avgqci[i][j][k] =  grid->meanpx[i][j][k]*qci[i+1][j][k]
							  +grid->meanmx[i][j][k]*qci[i-1][j][k]
							  +grid->meanpy[i][j][k]*qci[i][j+1][k]
							  +grid->meanmy[i][j][k]*qci[i][j-1][k]
							  +grid->meanpz[i][j][k]*qci[i][j][k+1]
							  +grid->meanmz[i][j][k]*qci[i][j][k-1];
					avgqcn[i][j][k] =  grid->meanpx[i][j][k]*qcn[i+1][j][k]
							  +grid->meanmx[i][j][k]*qcn[i-1][j][k]
							  +grid->meanpy[i][j][k]*qcn[i][j+1][k]
							  +grid->meanmy[i][j][k]*qcn[i][j-1][k]
							  +grid->meanpz[i][j][k]*qcn[i][j][k+1]
							  +grid->meanmz[i][j][k]*qcn[i][j][k-1];
					avgqsx[i][j][k] = grid->meanpx[i][j][k]*(vxbx[i+1][j][k]-vexbx[i+1][j][k]-mhd->rho[i+1][j][k]*mhd->gravx[i+1][j][k]-qsdx[i+1][j][k])
							 +grid->meanmx[i][j][k]*(vxbx[i-1][j][k]-vexbx[i-1][j][k]-mhd->rho[i-1][j][k]*mhd->gravx[i-1][j][k]-qsdx[i-1][j][k])
							 +grid->meanpy[i][j][k]*(vxbx[i][j+1][k]-vexbx[i][j+1][k]-mhd->rho[i][j+1][k]*mhd->gravx[i][j+1][k]-qsdx[i][j+1][k])
							 +grid->meanmy[i][j][k]*(vxbx[i][j-1][k]-vexbx[i][j-1][k]-mhd->rho[i][j-1][k]*mhd->gravx[i][j-1][k]-qsdx[i][j-1][k])
							 +grid->meanpz[i][j][k]*(vxbx[i][j][k+1]-vexbx[i][j][k+1]-mhd->rho[i][j][k+1]*mhd->gravx[i][j][k+1]-qsdx[i][j][k+1])
							 +grid->meanmz[i][j][k]*(vxbx[i][j][k-1]-vexbx[i][j][k-1]-mhd->rho[i][j][k-1]*mhd->gravx[i][j][k-1]-qsdx[i][j][k-1])
							 +grid->meanpx[i][j][k]*memd*mhd->rho[i+1][j][k]*mhd->zd[i+1][j][k]*rhoeinv[i+1][j][k]*(mhd->rhoe[i+1][j][k]*mhd->gravx[i+1][j][k]+qsex[i+1][j][k])
							 +grid->meanmx[i][j][k]*memd*mhd->rho[i-1][j][k]*mhd->zd[i-1][j][k]*rhoeinv[i-1][j][k]*(mhd->rhoe[i-1][j][k]*mhd->gravx[i-1][j][k]+qsex[i-1][j][k])
							 +grid->meanpy[i][j][k]*memd*mhd->rho[i][j+1][k]*mhd->zd[i][j+1][k]*rhoeinv[i][j+1][k]*(mhd->rhoe[i][j+1][k]*mhd->gravx[i][j+1][k]+qsex[i][j+1][k])
							 +grid->meanmy[i][j][k]*memd*mhd->rho[i][j-1][k]*mhd->zd[i][j-1][k]*rhoeinv[i][j-1][k]*(mhd->rhoe[i][j-1][k]*mhd->gravx[i][j-1][k]+qsex[i][j-1][k])
							 +grid->meanpz[i][j][k]*memd*mhd->rho[i][j][k+1]*mhd->zd[i][j][k+1]*rhoeinv[i][j][k+1]*(mhd->rhoe[i][j][k+1]*mhd->gravx[i][j][k+1]+qsex[i][j][k+1])
							 +grid->meanmz[i][j][k]*memd*mhd->rho[i][j][k-1]*mhd->zd[i][j][k-1]*rhoeinv[i][j][k-1]*(mhd->rhoe[i][j][k-1]*mhd->gravx[i][j][k-1]+qsex[i][j][k-1]);
					avgqsy[i][j][k] = grid->meanpx[i][j][k]*(vxby[i+1][j][k]-vexby[i+1][j][k]-mhd->rho[i+1][j][k]*mhd->gravy[i+1][j][k]-qsdy[i+1][j][k])
							 +grid->meanmx[i][j][k]*(vxby[i-1][j][k]-vexby[i-1][j][k]-mhd->rho[i-1][j][k]*mhd->gravy[i-1][j][k]-qsdy[i-1][j][k])
							 +grid->meanpy[i][j][k]*(vxby[i][j+1][k]-vexby[i][j+1][k]-mhd->rho[i][j+1][k]*mhd->gravy[i][j+1][k]-qsdy[i][j+1][k])
							 +grid->meanmy[i][j][k]*(vxby[i][j-1][k]-vexby[i][j-1][k]-mhd->rho[i][j-1][k]*mhd->gravy[i][j-1][k]-qsdy[i][j-1][k])
							 +grid->meanpz[i][j][k]*(vxby[i][j][k+1]-vexby[i][j][k+1]-mhd->rho[i][j][k+1]*mhd->gravy[i][j][k+1]-qsdy[i][j][k+1])
							 +grid->meanmz[i][j][k]*(vxby[i][j][k-1]-vexby[i][j][k-1]-mhd->rho[i][j][k-1]*mhd->gravy[i][j][k-1]-qsdy[i][j][k-1])
							 +grid->meanpx[i][j][k]*memd*mhd->rho[i+1][j][k]*mhd->zd[i+1][j][k]*rhoeinv[i+1][j][k]*(mhd->rhoe[i+1][j][k]*mhd->gravy[i+1][j][k]+qsey[i+1][j][k])
							 +grid->meanmx[i][j][k]*memd*mhd->rho[i-1][j][k]*mhd->zd[i-1][j][k]*rhoeinv[i-1][j][k]*(mhd->rhoe[i-1][j][k]*mhd->gravy[i-1][j][k]+qsey[i-1][j][k])
							 +grid->meanpy[i][j][k]*memd*mhd->rho[i][j+1][k]*mhd->zd[i][j+1][k]*rhoeinv[i][j+1][k]*(mhd->rhoe[i][j+1][k]*mhd->gravy[i][j+1][k]+qsey[i][j+1][k])
							 +grid->meanmy[i][j][k]*memd*mhd->rho[i][j-1][k]*mhd->zd[i][j-1][k]*rhoeinv[i][j-1][k]*(mhd->rhoe[i][j-1][k]*mhd->gravy[i][j-1][k]+qsey[i][j-1][k])
							 +grid->meanpz[i][j][k]*memd*mhd->rho[i][j][k+1]*mhd->zd[i][j][k+1]*rhoeinv[i][j][k+1]*(mhd->rhoe[i][j][k+1]*mhd->gravy[i][j][k+1]+qsey[i][j][k+1])
							 +grid->meanmz[i][j][k]*memd*mhd->rho[i][j][k-1]*mhd->zd[i][j][k-1]*rhoeinv[i][j][k-1]*(mhd->rhoe[i][j][k-1]*mhd->gravy[i][j][k-1]+qsey[i][j][k-1]);
					avgqsz[i][j][k] = grid->meanpx[i][j][k]*(vxbz[i+1][j][k]-vexbz[i+1][j][k]-mhd->rho[i+1][j][k]*mhd->gravz[i+1][j][k]-qsdz[i+1][j][k])
							 +grid->meanmx[i][j][k]*(vxbz[i-1][j][k]-vexbz[i-1][j][k]-mhd->rho[i-1][j][k]*mhd->gravz[i-1][j][k]-qsdz[i-1][j][k])
							 +grid->meanpy[i][j][k]*(vxbz[i][j+1][k]-vexbz[i][j+1][k]-mhd->rho[i][j+1][k]*mhd->gravz[i][j+1][k]-qsdz[i][j+1][k])
							 +grid->meanmy[i][j][k]*(vxbz[i][j-1][k]-vexbz[i][j-1][k]-mhd->rho[i][j-1][k]*mhd->gravz[i][j-1][k]-qsdz[i][j-1][k])
							 +grid->meanpz[i][j][k]*(vxbz[i][j][k+1]-vexbz[i][j][k+1]-mhd->rho[i][j][k+1]*mhd->gravz[i][j][k+1]-qsdz[i][j][k+1])
							 +grid->meanmz[i][j][k]*(vxbz[i][j][k-1]-vexbz[i][j][k-1]-mhd->rho[i][j][k-1]*mhd->gravz[i][j][k-1]-qsdz[i][j][k-1])
							 +grid->meanpx[i][j][k]*memd*mhd->rho[i+1][j][k]*mhd->zd[i+1][j][k]*rhoeinv[i+1][j][k]*(mhd->rhoe[i+1][j][k]*mhd->gravz[i+1][j][k]+qsez[i+1][j][k])
							 +grid->meanmx[i][j][k]*memd*mhd->rho[i-1][j][k]*mhd->zd[i-1][j][k]*rhoeinv[i-1][j][k]*(mhd->rhoe[i-1][j][k]*mhd->gravz[i-1][j][k]+qsez[i-1][j][k])
							 +grid->meanpy[i][j][k]*memd*mhd->rho[i][j+1][k]*mhd->zd[i][j+1][k]*rhoeinv[i][j+1][k]*(mhd->rhoe[i][j+1][k]*mhd->gravz[i][j+1][k]+qsez[i][j+1][k])
							 +grid->meanmy[i][j][k]*memd*mhd->rho[i][j-1][k]*mhd->zd[i][j-1][k]*rhoeinv[i][j-1][k]*(mhd->rhoe[i][j-1][k]*mhd->gravz[i][j-1][k]+qsez[i][j-1][k])
							 +grid->meanpz[i][j][k]*memd*mhd->rho[i][j][k+1]*mhd->zd[i][j][k+1]*rhoeinv[i][j][k+1]*(mhd->rhoe[i][j][k+1]*mhd->gravz[i][j][k+1]+qsez[i][j][k+1])
							 +grid->meanmz[i][j][k]*memd*mhd->rho[i][j][k-1]*mhd->zd[i][j][k-1]*rhoeinv[i][j][k-1]*(mhd->rhoe[i][j][k-1]*mhd->gravz[i][j][k-1]+qsez[i][j][k-1]);
					avgqsix[i][j][k] = grid->meanpx[i][j][k]*(ivixbx[i+1][j][k]-ivexbx[i+1][j][k]+mhd->rhoi[i+1][j][k]*mhd->gravx[i+1][j][k]+qsix[i+1][j][k])
							  +grid->meanmx[i][j][k]*(ivixbx[i-1][j][k]-ivexbx[i-1][j][k]+mhd->rhoi[i-1][j][k]*mhd->gravx[i-1][j][k]+qsix[i-1][j][k])
							  +grid->meanpy[i][j][k]*(ivixbx[i][j+1][k]-ivexbx[i][j+1][k]+mhd->rhoi[i][j+1][k]*mhd->gravx[i][j+1][k]+qsix[i][j+1][k])
							  +grid->meanmy[i][j][k]*(ivixbx[i][j-1][k]-ivexbx[i][j-1][k]+mhd->rhoi[i][j-1][k]*mhd->gravx[i][j-1][k]+qsix[i][j-1][k])
							  +grid->meanpz[i][j][k]*(ivixbx[i][j][k+1]-ivexbx[i][j][k+1]+mhd->rhoi[i][j][k+1]*mhd->gravx[i][j][k+1]+qsix[i][j][k+1])
							  +grid->meanmz[i][j][k]*(ivixbx[i][j][k-1]-ivexbx[i][j][k-1]+mhd->rhoi[i][j][k-1]*mhd->gravx[i][j][k-1]+qsix[i][j][k-1])
							  +grid->meanpx[i][j][k]*memi*mhd->rhoi[i+1][j][k]*mhd->zi[i+1][j][k]*rhoeinv[i+1][j][k]*(mhd->rhoe[i+1][j][k]*mhd->gravx[i+1][j][k]+qsex[i+1][j][k])
							  +grid->meanmx[i][j][k]*memi*mhd->rhoi[i-1][j][k]*mhd->zi[i-1][j][k]*rhoeinv[i-1][j][k]*(mhd->rhoe[i-1][j][k]*mhd->gravx[i-1][j][k]+qsex[i-1][j][k])
							  +grid->meanpy[i][j][k]*memi*mhd->rhoi[i][j+1][k]*mhd->zi[i][j+1][k]*rhoeinv[i][j+1][k]*(mhd->rhoe[i][j+1][k]*mhd->gravx[i][j+1][k]+qsex[i][j+1][k])
							  +grid->meanmy[i][j][k]*memi*mhd->rhoi[i][j-1][k]*mhd->zi[i][j-1][k]*rhoeinv[i][j-1][k]*(mhd->rhoe[i][j-1][k]*mhd->gravx[i][j-1][k]+qsex[i][j-1][k])
							  +grid->meanpz[i][j][k]*memi*mhd->rhoi[i][j][k+1]*mhd->zi[i][j][k+1]*rhoeinv[i][j][k+1]*(mhd->rhoe[i][j][k+1]*mhd->gravx[i][j][k+1]+qsex[i][j][k+1])
							  +grid->meanmz[i][j][k]*memi*mhd->rhoi[i][j][k-1]*mhd->zi[i][j][k-1]*rhoeinv[i][j][k-1]*(mhd->rhoe[i][j][k-1]*mhd->gravx[i][j][k-1]+qsex[i][j][k-1]);
					avgqsiy[i][j][k] = grid->meanpx[i][j][k]*(ivixby[i+1][j][k]-ivexby[i+1][j][k]+mhd->rhoi[i+1][j][k]*mhd->gravy[i+1][j][k]+qsiy[i+1][j][k])
							  +grid->meanmx[i][j][k]*(ivixby[i-1][j][k]-ivexby[i-1][j][k]+mhd->rhoi[i-1][j][k]*mhd->gravy[i-1][j][k]+qsiy[i-1][j][k])
							  +grid->meanpy[i][j][k]*(ivixby[i][j+1][k]-ivexby[i][j+1][k]+mhd->rhoi[i][j+1][k]*mhd->gravy[i][j+1][k]+qsiy[i][j+1][k])
							  +grid->meanmy[i][j][k]*(ivixby[i][j-1][k]-ivexby[i][j-1][k]+mhd->rhoi[i][j-1][k]*mhd->gravy[i][j-1][k]+qsiy[i][j-1][k])
							  +grid->meanpz[i][j][k]*(ivixby[i][j][k+1]-ivexby[i][j][k+1]+mhd->rhoi[i][j][k+1]*mhd->gravy[i][j][k+1]+qsiy[i][j][k+1])
							  +grid->meanmz[i][j][k]*(ivixby[i][j][k-1]-ivexby[i][j][k-1]+mhd->rhoi[i][j][k-1]*mhd->gravy[i][j][k-1]+qsiy[i][j][k-1])
							  +grid->meanpx[i][j][k]*memi*mhd->rhoi[i+1][j][k]*mhd->zi[i+1][j][k]*rhoeinv[i+1][j][k]*(mhd->rhoe[i+1][j][k]*mhd->gravy[i+1][j][k]+qsey[i+1][j][k])
							  +grid->meanmx[i][j][k]*memi*mhd->rhoi[i-1][j][k]*mhd->zi[i-1][j][k]*rhoeinv[i-1][j][k]*(mhd->rhoe[i-1][j][k]*mhd->gravy[i-1][j][k]+qsey[i-1][j][k])
							  +grid->meanpy[i][j][k]*memi*mhd->rhoi[i][j+1][k]*mhd->zi[i][j+1][k]*rhoeinv[i][j+1][k]*(mhd->rhoe[i][j+1][k]*mhd->gravy[i][j+1][k]+qsey[i][j+1][k])
							  +grid->meanmy[i][j][k]*memi*mhd->rhoi[i][j-1][k]*mhd->zi[i][j-1][k]*rhoeinv[i][j-1][k]*(mhd->rhoe[i][j-1][k]*mhd->gravy[i][j-1][k]+qsey[i][j-1][k])
							  +grid->meanpz[i][j][k]*memi*mhd->rhoi[i][j][k+1]*mhd->zi[i][j][k+1]*rhoeinv[i][j][k+1]*(mhd->rhoe[i][j][k+1]*mhd->gravy[i][j][k+1]+qsey[i][j][k+1])
							  +grid->meanmz[i][j][k]*memi*mhd->rhoi[i][j][k-1]*mhd->zi[i][j][k-1]*rhoeinv[i][j][k-1]*(mhd->rhoe[i][j][k-1]*mhd->gravy[i][j][k-1]+qsey[i][j][k-1]);
					avgqsiz[i][j][k] = grid->meanpx[i][j][k]*(ivixbz[i+1][j][k]-ivexbz[i+1][j][k]+mhd->rhoi[i+1][j][k]*mhd->gravz[i+1][j][k]+qsiz[i+1][j][k])
							  +grid->meanmx[i][j][k]*(ivixbz[i-1][j][k]-ivexbz[i-1][j][k]+mhd->rhoi[i-1][j][k]*mhd->gravz[i-1][j][k]+qsiz[i-1][j][k])
							  +grid->meanpy[i][j][k]*(ivixbz[i][j+1][k]-ivexbz[i][j+1][k]+mhd->rhoi[i][j+1][k]*mhd->gravz[i][j+1][k]+qsiz[i][j+1][k])
							  +grid->meanmy[i][j][k]*(ivixbz[i][j-1][k]-ivexbz[i][j-1][k]+mhd->rhoi[i][j-1][k]*mhd->gravz[i][j-1][k]+qsiz[i][j-1][k])
							  +grid->meanpz[i][j][k]*(ivixbz[i][j][k+1]-ivexbz[i][j][k+1]+mhd->rhoi[i][j][k+1]*mhd->gravz[i][j][k+1]+qsiz[i][j][k+1])
							  +grid->meanmz[i][j][k]*(ivixbz[i][j][k-1]-ivexbz[i][j][k-1]+mhd->rhoi[i][j][k-1]*mhd->gravz[i][j][k-1]+qsiz[i][j][k-1])
							  +grid->meanpx[i][j][k]*memi*mhd->rhoi[i+1][j][k]*mhd->zi[i+1][j][k]*rhoeinv[i+1][j][k]*(mhd->rhoe[i+1][j][k]*mhd->gravz[i+1][j][k]+qsez[i+1][j][k])
							  +grid->meanmx[i][j][k]*memi*mhd->rhoi[i-1][j][k]*mhd->zi[i-1][j][k]*rhoeinv[i-1][j][k]*(mhd->rhoe[i-1][j][k]*mhd->gravz[i-1][j][k]+qsez[i-1][j][k])
							  +grid->meanpy[i][j][k]*memi*mhd->rhoi[i][j+1][k]*mhd->zi[i][j+1][k]*rhoeinv[i][j+1][k]*(mhd->rhoe[i][j+1][k]*mhd->gravz[i][j+1][k]+qsez[i][j+1][k])
							  +grid->meanmy[i][j][k]*memi*mhd->rhoi[i][j-1][k]*mhd->zi[i][j-1][k]*rhoeinv[i][j-1][k]*(mhd->rhoe[i][j-1][k]*mhd->gravz[i][j-1][k]+qsez[i][j-1][k])
							  +grid->meanpz[i][j][k]*memi*mhd->rhoi[i][j][k+1]*mhd->zi[i][j][k+1]*rhoeinv[i][j][k+1]*(mhd->rhoe[i][j][k+1]*mhd->gravz[i][j][k+1]+qsez[i][j][k+1])
							  +grid->meanmz[i][j][k]*memi*mhd->rhoi[i][j][k-1]*mhd->zi[i][j][k-1]*rhoeinv[i][j][k-1]*(mhd->rhoe[i][j][k-1]*mhd->gravz[i][j][k-1]+qsez[i][j][k-1]);
					avgqsnx[i][j][k] = grid->meanpx[i][j][k]*(mhd->rhon[i+1][j][k]*mhd->gravx[i+1][j][k]+qsnx[i+1][j][k])
							  +grid->meanmx[i][j][k]*(mhd->rhon[i-1][j][k]*mhd->gravx[i-1][j][k]+qsnx[i-1][j][k])
							  +grid->meanpy[i][j][k]*(mhd->rhon[i][j+1][k]*mhd->gravx[i][j+1][k]+qsnx[i][j+1][k])
							  +grid->meanmy[i][j][k]*(mhd->rhon[i][j-1][k]*mhd->gravx[i][j-1][k]+qsnx[i][j-1][k])
							  +grid->meanpz[i][j][k]*(mhd->rhon[i][j][k+1]*mhd->gravx[i][j][k+1]+qsnx[i][j][k+1])
							  +grid->meanmz[i][j][k]*(mhd->rhon[i][j][k-1]*mhd->gravx[i][j][k-1]+qsnx[i][j][k-1]);
					avgqsny[i][j][k] = grid->meanpx[i][j][k]*(mhd->rhon[i+1][j][k]*mhd->gravy[i+1][j][k]+qsny[i+1][j][k])
							  +grid->meanmx[i][j][k]*(mhd->rhon[i-1][j][k]*mhd->gravy[i-1][j][k]+qsny[i-1][j][k])
							  +grid->meanpy[i][j][k]*(mhd->rhon[i][j+1][k]*mhd->gravy[i][j+1][k]+qsny[i][j+1][k])
							  +grid->meanmy[i][j][k]*(mhd->rhon[i][j-1][k]*mhd->gravy[i][j-1][k]+qsny[i][j-1][k])
							  +grid->meanpz[i][j][k]*(mhd->rhon[i][j][k+1]*mhd->gravy[i][j][k+1]+qsny[i][j][k+1])
							  +grid->meanmz[i][j][k]*(mhd->rhon[i][j][k-1]*mhd->gravy[i][j][k-1]+qsny[i][j][k-1]);
					avgqsnz[i][j][k] = grid->meanpx[i][j][k]*(mhd->rhon[i+1][j][k]*mhd->gravz[i+1][j][k]+qsnz[i+1][j][k])
							  +grid->meanmx[i][j][k]*(mhd->rhon[i-1][j][k]*mhd->gravz[i-1][j][k]+qsnz[i-1][j][k])
							  +grid->meanpy[i][j][k]*(mhd->rhon[i][j+1][k]*mhd->gravz[i][j+1][k]+qsnz[i][j+1][k])
							  +grid->meanmy[i][j][k]*(mhd->rhon[i][j-1][k]*mhd->gravz[i][j-1][k]+qsnz[i][j-1][k])
							  +grid->meanpz[i][j][k]*(mhd->rhon[i][j][k+1]*mhd->gravz[i][j][k+1]+qsnz[i][j][k+1])
							  +grid->meanmz[i][j][k]*(mhd->rhon[i][j][k-1]*mhd->gravz[i][j][k-1]+qsnz[i][j][k-1]);
					avgqed[i][j][k] =  grid->meanpx[i][j][k]*qed[i+1][j][k]
							  +grid->meanmx[i][j][k]*qed[i-1][j][k]
							  +grid->meanpy[i][j][k]*qed[i][j+1][k]
							  +grid->meanmy[i][j][k]*qed[i][j-1][k]
							  +grid->meanpz[i][j][k]*qed[i][j][k+1]
							  +grid->meanmz[i][j][k]*qed[i][j][k-1];
					avgqei[i][j][k] =  grid->meanpx[i][j][k]*qei[i+1][j][k]
							  +grid->meanmx[i][j][k]*qei[i-1][j][k]
							  +grid->meanpy[i][j][k]*qei[i][j+1][k]
							  +grid->meanmy[i][j][k]*qei[i][j-1][k]
							  +grid->meanpz[i][j][k]*qei[i][j][k+1]
							  +grid->meanmz[i][j][k]*qei[i][j][k-1];
					avgqee[i][j][k] =  grid->meanpx[i][j][k]*qee[i+1][j][k]
							  +grid->meanmx[i][j][k]*qee[i-1][j][k]
							  +grid->meanpy[i][j][k]*qee[i][j+1][k]
							  +grid->meanmy[i][j][k]*qee[i][j-1][k]
							  +grid->meanpz[i][j][k]*qee[i][j][k+1]
							  +grid->meanmz[i][j][k]*qee[i][j][k-1];
					avgqen[i][j][k] =  grid->meanpx[i][j][k]*qen[i+1][j][k]
							  +grid->meanmx[i][j][k]*qen[i-1][j][k]
							  +grid->meanpy[i][j][k]*qen[i][j+1][k]
							  +grid->meanmy[i][j][k]*qen[i][j-1][k]
							  +grid->meanpz[i][j][k]*qen[i][j][k+1]
							  +grid->meanmz[i][j][k]*qen[i][j][k-1];
					//  Calculate Derivative Terms
					//Densities
					drho[i][j][k]= grid->difx[i][j][k]*(mhd->sx[i+1][j][k]-mhd->sx[i-1][j][k])
						      +grid->dify[i][j][k]*(mhd->sy[i][j+1][k]-mhd->sy[i][j-1][k])
						      +grid->difz[i][j][k]*(mhd->sz[i][j][k+1]-mhd->sz[i][j][k-1]);
					drhoi[i][j][k]= grid->difx[i][j][k]*(mhd->six[i+1][j][k]-mhd->six[i-1][j][k])
						       +grid->dify[i][j][k]*(mhd->siy[i][j+1][k]-mhd->siy[i][j-1][k])
						       +grid->difz[i][j][k]*(mhd->siz[i][j][k+1]-mhd->siz[i][j][k-1]);
					drhoe[i][j][k]= grid->difx[i][j][k]*(mhd->sex[i+1][j][k]-mhd->sex[i-1][j][k])
						       +grid->dify[i][j][k]*(mhd->sey[i][j+1][k]-mhd->sey[i][j-1][k])
						       +grid->difz[i][j][k]*(mhd->sez[i][j][k+1]-mhd->sez[i][j][k-1]);
					drhon[i][j][k]= grid->difx[i][j][k]*(mhd->snx[i+1][j][k]-mhd->snx[i-1][j][k])
						       +grid->dify[i][j][k]*(mhd->sny[i][j+1][k]-mhd->sny[i][j-1][k])
						       +grid->difz[i][j][k]*(mhd->snz[i][j][k+1]-mhd->snz[i][j][k-1]);
					//Momentums (note the use of spatial average for the pe coefficients)
					dsx[i][j][k]= grid->difx[i][j][k]*(rhovvxx[i+1][j][k]-rhovvxx[i-1][j][k]+mhd->p[i+1][j][k]-mhd->p[i-1][j][k])
						     +grid->dify[i][j][k]*(rhovvxy[i][j+1][k]-rhovvxy[i][j-1][k])
						     +grid->difz[i][j][k]*(rhovvxz[i][j][k+1]-rhovvxz[i][j][k-1]);
//						     -grid->difx[i][j][k]*mhd->zd[i][j][k]*memd*mhd->rho[i][j][k]*rhoeinv[i][j][k]*(mhd->pe[i+1][j][k]-mhd->pe[i-1][j][k]);						     
					avgdsx[i][j][k]=-grid->difx[i][j][k]*avgped[i][j][k]*(mhd->pe[i+1][j][k]-mhd->pe[i-1][j][k]);						     
					dsy[i][j][k]= grid->dify[i][j][k]*(rhovvyy[i][j+1][k]-rhovvyy[i][j-1][k]+mhd->p[i][j+1][k]-mhd->p[i][j-1][k])
						     +grid->difx[i][j][k]*(rhovvxy[i+1][j][k]-rhovvxy[i-1][j][k])
						     +grid->difz[i][j][k]*(rhovvyz[i][j][k+1]-rhovvyz[i][j][k-1]);
//						     -grid->dify[i][j][k]*mhd->zd[i][j][k]*memd*mhd->rho[i][j][k]*rhoeinv[i][j][k]*(mhd->pe[i][j+1][k]-mhd->pe[i][j-1][k]);						     
					avgdsy[i][j][k]=-grid->dify[i][j][k]*avgped[i][j][k]*(mhd->pe[i][j+1][k]-mhd->pe[i][j-1][k]);
					dsz[i][j][k]= grid->difz[i][j][k]*(rhovvzz[i][j][k+1]-rhovvzz[i][j][k-1]+mhd->p[i][j][k+1]-mhd->p[i][j][k-1])
						     +grid->difx[i][j][k]*(rhovvxz[i+1][j][k]-rhovvxz[i-1][j][k])
						     +grid->dify[i][j][k]*(rhovvyz[i][j+1][k]-rhovvyz[i][j-1][k]);
//						     -grid->difz[i][j][k]*mhd->zd[i][j][k]*memd*mhd->rho[i][j][k]*rhoeinv[i][j][k]*(mhd->pe[i][j][k+1]-mhd->pe[i][j][k-1]);						     
					avgdsz[i][j][k]=-grid->difz[i][j][k]*avgped[i][j][k]*(mhd->pe[i][j][k+1]-mhd->pe[i][j][k-1]);
					dsix[i][j][k]= grid->difx[i][j][k]*(rhoivvxx[i+1][j][k]-rhoivvxx[i-1][j][k]+mhd->pi[i+1][j][k]-mhd->pi[i-1][j][k])
						      +grid->dify[i][j][k]*(rhoivvxy[i][j+1][k]-rhoivvxy[i][j-1][k])
						      +grid->difz[i][j][k]*(rhoivvxz[i][j][k+1]-rhoivvxz[i][j][k-1]);
//						      +grid->difx[i][j][k]*mhd->zi[i][j][k]*memi*mhd->rhoi[i][j][k]*rhoeinv[i][j][k]*(mhd->pe[i+1][j][k]-mhd->pe[i-1][j][k]);						     
					avgdsix[i][j][k]=+grid->difx[i][j][k]*avgpei[i][j][k]*(mhd->pe[i+1][j][k]-mhd->pe[i-1][j][k]);
					dsiy[i][j][k]= grid->dify[i][j][k]*(rhoivvyy[i][j+1][k]-rhoivvyy[i][j-1][k]+mhd->pi[i][j+1][k]-mhd->pi[i][j-1][k])
						      +grid->difx[i][j][k]*(rhoivvxy[i+1][j][k]-rhoivvxy[i-1][j][k])
						      +grid->difz[i][j][k]*(rhoivvyz[i][j][k+1]-rhoivvyz[i][j][k-1]);
//						      +grid->dify[i][j][k]*mhd->zi[i][j][k]*memi*mhd->rhoi[i][j][k]*rhoeinv[i][j][k]*(mhd->pe[i][j+1][k]-mhd->pe[i][j-1][k]);						     
					avgdsiy[i][j][k]=+grid->dify[i][j][k]*avgpei[i][j][k]*(mhd->pe[i][j+1][k]-mhd->pe[i][j-1][k]);
					dsiz[i][j][k]= grid->difz[i][j][k]*(rhoivvzz[i][j][k+1]-rhoivvzz[i][j][k-1]+mhd->pi[i][j][k+1]-mhd->pi[i][j][k-1])
						      +grid->difx[i][j][k]*(rhoivvxz[i+1][j][k]-rhoivvxz[i-1][j][k])
						      +grid->dify[i][j][k]*(rhoivvyz[i][j+1][k]-rhoivvyz[i][j-1][k]);
//						      +grid->difz[i][j][k]*mhd->zi[i][j][k]*memi*mhd->rhoi[i][j][k]*rhoeinv[i][j][k]*(mhd->pe[i][j][k+1]-mhd->pe[i][j][k-1]);						     
					avgdsiz[i][j][k]=+grid->difz[i][j][k]*avgpei[i][j][k]*(mhd->pe[i][j][k+1]-mhd->pe[i][j][k-1]);
					dsnx[i][j][k]= grid->difx[i][j][k]*(rhonvvxx[i+1][j][k]-rhonvvxx[i-1][j][k]+mhd->pn[i+1][j][k]-mhd->pn[i-1][j][k])
						      +grid->dify[i][j][k]*(rhonvvxy[i][j+1][k]-rhonvvxy[i][j-1][k])
						      +grid->difz[i][j][k]*(rhonvvxz[i][j][k+1]-rhonvvxz[i][j][k-1]);
					dsny[i][j][k]= grid->dify[i][j][k]*(rhonvvyy[i][j+1][k]-rhonvvyy[i][j-1][k]+mhd->pn[i][j+1][k]-mhd->pn[i][j-1][k])
						      +grid->difx[i][j][k]*(rhonvvxy[i+1][j][k]-rhonvvxy[i-1][j][k])
						      +grid->difz[i][j][k]*(rhonvvyz[i][j][k+1]-rhonvvyz[i][j][k-1]);
					dsnz[i][j][k]= grid->difz[i][j][k]*(rhonvvzz[i][j][k+1]-rhonvvzz[i][j][k-1]+mhd->pn[i][j][k+1]-mhd->pn[i][j][k-1])
						      +grid->difx[i][j][k]*(rhonvvxz[i+1][j][k]-rhonvvxz[i-1][j][k])
						      +grid->dify[i][j][k]*(rhonvvyz[i][j+1][k]-rhonvvyz[i][j-1][k]);
					//Pressures
					dp[i][j][k]= grid->difx[i][j][k]*(mhd->p[i+1][j][k]*vx[i+1][j][k]-mhd->p[i-1][j][k]*vx[i-1][j][k])
						    +grid->dify[i][j][k]*(mhd->p[i][j+1][k]*vy[i][j+1][k]-mhd->p[i][j-1][k]*vy[i][j-1][k])
						    +grid->difz[i][j][k]*(mhd->p[i][j][k+1]*vz[i][j][k+1]-mhd->p[i][j][k-1]*vz[i][j][k-1]);
//						    +(mhd->gamma[i][j][k]-1.)*mhd->p[i][j][k]*( grid->difx[i][j][k]*(vx[i+1][j][k]-vx[i-1][j][k])
//											   +grid->dify[i][j][k]*(vy[i][j+1][k]-vy[i][j-1][k])
//											   +grid->difz[i][j][k]*(vz[i][j][k+1]-vz[i][j][k-1]));
					avgdpd[i][j][k]=(mhd->gamma[i][j][k]-1.)*( grid->difx[i][j][k]*(vx[i+1][j][k]-vx[i-1][j][k])
									          +grid->dify[i][j][k]*(vy[i][j+1][k]-vy[i][j-1][k])
									          +grid->difz[i][j][k]*(vz[i][j][k+1]-vz[i][j][k-1]))
									   *spat*( mhd->p[i+1][j][k] + mhd->p[i-1][j][k]
									          +mhd->p[i][j+1][k] + mhd->p[i][j-1][k]
									          +mhd->p[i][j][k+1] + mhd->p[i][j][k-1]);
					dpi[i][j][k]= grid->difx[i][j][k]*(mhd->pi[i+1][j][k]*vix[i+1][j][k]-mhd->pi[i-1][j][k]*vix[i-1][j][k])
						     +grid->dify[i][j][k]*(mhd->pi[i][j+1][k]*viy[i][j+1][k]-mhd->pi[i][j-1][k]*viy[i][j-1][k])
						     +grid->difz[i][j][k]*(mhd->pi[i][j][k+1]*viz[i][j][k+1]-mhd->pi[i][j][k-1]*viz[i][j][k-1]);
//						     +(mhd->gammai[i][j][k]-1.)*mhd->pi[i][j][k]*( grid->difx[i][j][k]*(vix[i+1][j][k]-vix[i-1][j][k])
//											      +grid->dify[i][j][k]*(viy[i][j+1][k]-viy[i][j-1][k])
//											      +grid->difz[i][j][k]*(viz[i][j][k+1]-viz[i][j][k-1]));
					avgdpi[i][j][k]=(mhd->gammai[i][j][k]-1.)*( grid->difx[i][j][k]*(vix[i+1][j][k]-vix[i-1][j][k])
									           +grid->dify[i][j][k]*(viy[i][j+1][k]-viy[i][j-1][k])
									           +grid->difz[i][j][k]*(viz[i][j][k+1]-viz[i][j][k-1]))
									    *spat*( mhd->pi[i+1][j][k] + mhd->pi[i-1][j][k]
									           +mhd->pi[i][j+1][k] + mhd->pi[i][j-1][k]
									           +mhd->pi[i][j][k+1] + mhd->pi[i][j][k-1]);
					dpe[i][j][k]= grid->difx[i][j][k]*(mhd->pe[i+1][j][k]*vex[i+1][j][k]-mhd->pe[i-1][j][k]*vex[i-1][j][k])
						     +grid->dify[i][j][k]*(mhd->pe[i][j+1][k]*vey[i][j+1][k]-mhd->pe[i][j-1][k]*vey[i][j-1][k])
						     +grid->difz[i][j][k]*(mhd->pe[i][j][k+1]*vez[i][j][k+1]-mhd->pe[i][j][k-1]*vez[i][j][k-1]);
//						     +(mhd->gammae[i][j][k]-1.)*mhd->pe[i][j][k]*( grid->difx[i][j][k]*(vex[i+1][j][k]-vex[i-1][j][k])
//											      +grid->dify[i][j][k]*(vey[i][j+1][k]-vey[i][j-1][k])
//											      +grid->difz[i][j][k]*(vez[i][j][k+1]-vez[i][j][k-1]));
					avgdpe[i][j][k]=(mhd->gammae[i][j][k]-1.)*( grid->difx[i][j][k]*(vex[i+1][j][k]-vex[i-1][j][k])
									           +grid->dify[i][j][k]*(vey[i][j+1][k]-vey[i][j-1][k])
									           +grid->difz[i][j][k]*(vez[i][j][k+1]-vez[i][j][k-1]))
									    *spat*( mhd->pe[i+1][j][k] + mhd->pe[i-1][j][k]
									           +mhd->pe[i][j+1][k] + mhd->pe[i][j-1][k]
									           +mhd->pe[i][j][k+1] + mhd->pe[i][j][k-1]);
					dpn[i][j][k]= grid->difx[i][j][k]*(mhd->pn[i+1][j][k]*vnx[i+1][j][k]-mhd->pn[i-1][j][k]*vnx[i-1][j][k])
						     +grid->dify[i][j][k]*(mhd->pn[i][j+1][k]*vny[i][j+1][k]-mhd->pn[i][j-1][k]*vny[i][j-1][k])
						     +grid->difz[i][j][k]*(mhd->pn[i][j][k+1]*vnz[i][j][k+1]-mhd->pn[i][j][k-1]*vnz[i][j][k-1]);
//						     +(mhd->gamman[i][j][k]-1.)*mhd->pn[i][j][k]*( grid->difx[i][j][k]*(vnx[i+1][j][k]-vnx[i-1][j][k])
//											      +grid->dify[i][j][k]*(vny[i][j+1][k]-vny[i][j-1][k])
//											      +grid->difz[i][j][k]*(vnz[i][j][k+1]-vnz[i][j][k-1]));
					avgdpn[i][j][k]=(mhd->gamman[i][j][k]-1.)*( grid->difx[i][j][k]*(vnx[i+1][j][k]-vnx[i-1][j][k])
									           +grid->dify[i][j][k]*(vny[i][j+1][k]-vny[i][j-1][k])
									           +grid->difz[i][j][k]*(vnz[i][j][k+1]-vnz[i][j][k-1]))
									    *spat*( mhd->pn[i+1][j][k] + mhd->pn[i-1][j][k]
									           +mhd->pn[i][j+1][k] + mhd->pn[i][j-1][k]
									           +mhd->pn[i][j][k+1] + mhd->pn[i][j][k-1]);
					//Induction (grad pe term turned off to supress DIV(B)
					dbx[i][j][k]=
					             +grid->dify[i][j][k]*(bvexbz[i][j+1][k]-bvexbz[i][j-1][k])
						     -grid->difz[i][j][k]*(bvexby[i][j][k+1]-bvexby[i][j][k-1])
//						     +grid->dify[i][j][k]*(bvixbz[i][j+1][k]-bvixbz[i][j-1][k])
//						     -grid->difz[i][j][k]*(bvixby[i][j][k+1]-bvixby[i][j][k-1])
//						     -grid->dify[i][j][k]*(bvxbz[i][j+1][k]-bvxbz[i][j-1][k])
//						     +grid->difz[i][j][k]*(bvxby[i][j][k+1]-bvxby[i][j][k-1])
//						     -merhoe[i][j][k]*einv*( grid->dify[i][j][k]*(mhd->pe[i][j+1][k]-mhd->pe[i][j-1][k])*grid->difz[i][j][k]*(mhd->rhoe[i][j][k+1]-mhd->rhoe[i][j][k-1])
//									    -grid->dify[i][j][k]*(mhd->rhoe[i][j+1][k]-mhd->rhoe[i][j-1][k])*grid->difz[i][j][k]*(mhd->pe[i][j][k+1]-mhd->pe[i][j][k-1]))
						     -mee*( grid->dify[i][j][k]*(mhd->gravz[i][j+1][k]-mhd->gravz[i][j-1][k]+qsez[i][j+1][k]*rhoeinv[i][j+1][k]-qsez[i][j-1][k]*rhoeinv[i][j-1][k])
							   -grid->difz[i][j][k]*(mhd->gravy[i][j][k+1]-mhd->gravy[i][j][k-1]+qsey[i][j][k+1]*rhoeinv[i][j][k+1]-qsey[i][j][k-1]*rhoeinv[i][j][k-1]));
					dby[i][j][k]=
					             +grid->difz[i][j][k]*(bvexbx[i][j][k+1]-bvexbx[i][j][k-1])
						     -grid->difx[i][j][k]*(bvexbz[i+1][j][k]-bvexbz[i-1][j][k])
//					             +grid->difz[i][j][k]*(bvixbx[i][j][k+1]-bvixbx[i][j][k-1])
//						     -grid->difx[i][j][k]*(bvixbz[i+1][j][k]-bvixbz[i-1][j][k])
//					             -grid->difz[i][j][k]*(bvxbx[i][j][k+1]-bvxbx[i][j][k-1])
//						     +grid->difx[i][j][k]*(bvxbz[i+1][j][k]-bvxbz[i-1][j][k])
//						     -merhoe[i][j][k]*einv*( grid->difx[i][j][k]*(mhd->rhoe[i+1][j][k]-mhd->rhoe[i-1][j][k])*grid->difz[i][j][k]*(mhd->pe[i][j][k+1]-mhd->pe[i][j][k-1])
//									    -grid->difx[i][j][k]*(mhd->pe[i+1][j][k]-mhd->pe[i-1][j][k])*grid->difz[i][j][k]*(mhd->rhoe[i][j][k+1]-mhd->rhoe[i][j][k-1]))
						     -mee*( grid->difz[i][j][k]*(mhd->gravx[i][j][k+1]-mhd->gravx[i][j][k-1]+qsex[i][j][k+1]*rhoeinv[i][j][k+1]-qsex[i][j][k-1]*rhoeinv[i][j][k-1])
							   -grid->difx[i][j][k]*(mhd->gravz[i+1][j][k]-mhd->gravz[i-1][j][k]+qsez[i+1][j][k]*rhoeinv[i+1][j][k]-qsez[i-1][j][k]*rhoeinv[i-1][j][k]));
					dbz[i][j][k]=
					             +grid->difx[i][j][k]*(bvexby[i+1][j][k]-bvexby[i-1][j][k])
						     -grid->dify[i][j][k]*(bvexbx[i][j+1][k]-bvexbx[i][j-1][k])
//					             +grid->difx[i][j][k]*(bvixby[i+1][j][k]-bvixby[i-1][j][k])
//						     -grid->dify[i][j][k]*(bvixbx[i][j+1][k]-bvixbx[i][j-1][k])
//					             -grid->difx[i][j][k]*(bvxby[i+1][j][k]-bvxby[i-1][j][k])
//						     +grid->dify[i][j][k]*(bvxbx[i][j+1][k]-bvxbx[i][j-1][k])
//						     -merhoe[i][j][k]*einv*( grid->difx[i][j][k]*(mhd->pe[i+1][j][k]-mhd->pe[i-1][j][k])*grid->dify[i][j][k]*(mhd->rhoe[i][j+1][k]-mhd->rhoe[i][j-1][k])
//									    -grid->difx[i][j][k]*(mhd->rhoe[i+1][j][k]-mhd->rhoe[i-1][j][k])*grid->dify[i][j][k]*(mhd->pe[i][j+1][k]-mhd->pe[i][j-1][k]))
						     -mee*( grid->difx[i][j][k]*(mhd->gravy[i+1][j][k]-mhd->gravy[i-1][j][k]+qsey[i+1][j][k]*rhoeinv[i+1][j][k]-qsey[i-1][j][k]*rhoeinv[i-1][j][k])
						     	   -grid->dify[i][j][k]*(mhd->gravx[i][j+1][k]-mhd->gravx[i][j-1][k]+qsex[i][j+1][k]*rhoeinv[i][j+1][k]-qsex[i][j-1][k]*rhoeinv[i][j-1][k]));
				}
			 }
		}
//		if (tid == 0)
//		{
//			printf(".......%04.2f [s] -----\n",(omp_get_wtime()-stime)/TIMESCALE);
//			stime=omp_get_wtime();
//			printf("----- First Timestep");
//		}
		//First step u(t,i)=u(t-1,i)-dt/(2dx)*(u(t,i+1)-u(t,i-1))+(alpha*dt/2)*(u(t,i+1)-u(t,i-1))
//		#pragma omp barrier
		#pragma omp for
		for(i=1;i<grid->nxm1;i++)
		{
			for(j=1;j<grid->nym1;j++)
			{
				for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
				{
					//printf("gchk=%d i=%d j=%d k=%d\n",gchk,i,j,k);
					//  Density Equations
					mhd->rho[i][j][k]=mhd->rho[i][j][k]-grid->dt*drho[i][j][k];
					mhd->rhoi[i][j][k]=mhd->rhoi[i][j][k]-grid->dt*drhoi[i][j][k]+grid->dt*spat*avgqci[i][j][k];
					mhd->rhon[i][j][k]=mhd->rhon[i][j][k]-grid->dt*drhon[i][j][k]+grid->dt*spat*avgqcn[i][j][k];
					//  Momentum Equations
					mhd->sx[i][j][k]=mhd->sx[i][j][k]-grid->dt*dsx[i][j][k]-spat*grid->dt*avgqsx[i][j][k]
					                                 -grid->dt*avgdsx[i][j][k];
					mhd->sy[i][j][k]=mhd->sy[i][j][k]-grid->dt*dsy[i][j][k]-spat*grid->dt*avgqsy[i][j][k]
					                                 -grid->dt*avgdsy[i][j][k];
					mhd->sz[i][j][k]=mhd->sz[i][j][k]-grid->dt*dsz[i][j][k]-spat*grid->dt*avgqsz[i][j][k]
					                                 -grid->dt*avgdsz[i][j][k];
					mhd->six[i][j][k]=mhd->six[i][j][k]-grid->dt*dsix[i][j][k]+spat*grid->dt*avgqsix[i][j][k]
					                                   -grid->dt*avgdsix[i][j][k];
					mhd->siy[i][j][k]=mhd->siy[i][j][k]-grid->dt*dsiy[i][j][k]+spat*grid->dt*avgqsiy[i][j][k]
					                                   -grid->dt*avgdsiy[i][j][k];
					mhd->siz[i][j][k]=mhd->siz[i][j][k]-grid->dt*dsiz[i][j][k]+spat*grid->dt*avgqsiz[i][j][k]
					                                   -grid->dt*avgdsiz[i][j][k];
					mhd->snx[i][j][k]=mhd->snx[i][j][k]-grid->dt*dsnx[i][j][k]-spat*grid->dt*avgqsnx[i][j][k];
					mhd->sny[i][j][k]=mhd->sny[i][j][k]-grid->dt*dsny[i][j][k]-spat*grid->dt*avgqsny[i][j][k];
					mhd->snz[i][j][k]=mhd->snz[i][j][k]-grid->dt*dsnz[i][j][k]-spat*grid->dt*avgqsnz[i][j][k];
					//  Presure Equations
					mhd->p[i][j][k]=mhd->p[i][j][k]-grid->dt*dp[i][j][k]
								       +grid->dt*(mhd->gamma[i][j][k]-1.)*spat*avgqed[i][j][k]
			                                               -grid->dt*avgdpd[i][j][k];
					mhd->pi[i][j][k]=mhd->pi[i][j][k]-grid->dt*dpi[i][j][k]
									 +grid->dt*(mhd->gammai[i][j][k]-1.)*spat*avgqei[i][j][k]
			                                                 -grid->dt*avgdpi[i][j][k];
					mhd->pe[i][j][k]=mhd->pe[i][j][k]-grid->dt*dpe[i][j][k]
									 +(mhd->gammae[i][j][k]-1.)*spat*grid->dt*avgqee[i][j][k]
			                                                 -grid->dt*avgdpe[i][j][k];
					mhd->pn[i][j][k]=mhd->pn[i][j][k]-grid->dt*dpn[i][j][k]
									 +grid->dt*(mhd->gamman[i][j][k]-1.)*spat*avgqen[i][j][k]
			                                                 -grid->dt*avgdpn[i][j][k];
					//  Induction Equations
					mhd->bx[i][j][k]=mhd->bx[i][j][k]+grid->dt*dbx[i][j][k];
					mhd->by[i][j][k]=mhd->by[i][j][k]+grid->dt*dby[i][j][k];
					mhd->bz[i][j][k]=mhd->bz[i][j][k]+grid->dt*dbz[i][j][k];
				}
			}
		}
		//Update Electron Continuity (removed from previous for loop for parallelization due to rho reference.
		#pragma omp for
		for(i=1;i<grid->nxm1;i++)
		{
			for(j=1;j<grid->nym1;j++)
			{
				for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
				{
					mhd->rhoe[i][j][k]= (1-grid->econt)*mhd->me*(mhd->zi[i][j][k]*mhd->rhoi[i][j][k]*miinv-mhd->zd[i][j][k]*mhd->rho[i][j][k]*mdinv)
					                   +grid->econt*(mhd->rhoe[i][j][k]-grid->dt*drhoe[i][j][k]);
				}
			}
		}
		//Now we recalc our source terms on the centered grid (note j does not change)
//		if (tid == 0)
//		{
//			printf("....................%04.2f [s] -----\n",(omp_get_wtime()-stime)/TIMESCALE);
//			stime=omp_get_wtime();
//			printf("----- Recalc Source Terms");
//		}
//		#pragma omp barrier
		#pragma omp for
		for(i=1;i<grid->nxm1;i++)
		{
			for(j=1;j<grid->nym1;j++)
			{
				for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
				{
				//    Electron Momentum
				mhd->sex[i][j][k]=memi*mhd->zi[i][j][k]*mhd->six[i][j][k]-memd*mhd->zd[i][j][k]*mhd->sx[i][j][k]-mhd->me*mhd->jx[i][j][k]/mhd->e;
				mhd->sey[i][j][k]=memi*mhd->zi[i][j][k]*mhd->siy[i][j][k]-memd*mhd->zd[i][j][k]*mhd->sy[i][j][k]-mhd->me*mhd->jy[i][j][k]/mhd->e;
				mhd->sez[i][j][k]=memi*mhd->zi[i][j][k]*mhd->siz[i][j][k]-memd*mhd->zd[i][j][k]*mhd->sz[i][j][k]-mhd->me*mhd->jz[i][j][k]/mhd->e;
				//    Rho's
				srho[i][j][k]=mhd->rho[i][j][k];
				srhoi[i][j][k]=mhd->rhoi[i][j][k];
				srhoe[i][j][k]=mhd->rhoe[i][j][k];
				srhon[i][j][k]=mhd->rhon[i][j][k];
				rhoinv[i][j][k]=1./mhd->rho[i][j][k];
				rhoiinv[i][j][k]=1./mhd->rhoi[i][j][k];
				rhoeinv[i][j][k]=1./mhd->rhoe[i][j][k];
				rhoninv[i][j][k]=1./mhd->rhon[i][j][k];
				//      v's
				vx[i][j][k]=mhd->sx[i][j][k]*rhoinv[i][j][k];
				vy[i][j][k]=mhd->sy[i][j][k]*rhoinv[i][j][k];
				vz[i][j][k]=mhd->sz[i][j][k]*rhoinv[i][j][k];
				vix[i][j][k]=mhd->six[i][j][k]*rhoiinv[i][j][k];
				viy[i][j][k]=mhd->siy[i][j][k]*rhoiinv[i][j][k];
				viz[i][j][k]=mhd->siz[i][j][k]*rhoiinv[i][j][k];
				vex[i][j][k]=mhd->sex[i][j][k]*rhoeinv[i][j][k];
				vey[i][j][k]=mhd->sey[i][j][k]*rhoeinv[i][j][k];
				vez[i][j][k]=mhd->sez[i][j][k]*rhoeinv[i][j][k];
				vnx[i][j][k]=mhd->snx[i][j][k]*rhoninv[i][j][k];
				vny[i][j][k]=mhd->sny[i][j][k]*rhoninv[i][j][k];
				vnz[i][j][k]=mhd->snz[i][j][k]*rhoninv[i][j][k];
				//	vxb term
				vxbx[i][j][k]=mhd->zd[i][j][k]*mhd->e*mdinv*(mhd->sy[i][j][k]*mhd->bz[i][j][k]-mhd->sz[i][j][k]*mhd->by[i][j][k]);
				vxby[i][j][k]=mhd->zd[i][j][k]*mhd->e*mdinv*(mhd->sz[i][j][k]*mhd->bx[i][j][k]-mhd->sx[i][j][k]*mhd->bz[i][j][k]);
				vxbz[i][j][k]=mhd->zd[i][j][k]*mhd->e*mdinv*(mhd->sx[i][j][k]*mhd->by[i][j][k]-mhd->sy[i][j][k]*mhd->bx[i][j][k]);
				//	vexb term
				vexbx[i][j][k]=mhd->zd[i][j][k]*mhd->e*mdinv*mhd->rho[i][j][k]*(vey[i][j][k]*mhd->bz[i][j][k]-vez[i][j][k]*mhd->by[i][j][k]);
				vexby[i][j][k]=mhd->zd[i][j][k]*mhd->e*mdinv*mhd->rho[i][j][k]*(vez[i][j][k]*mhd->bx[i][j][k]-vex[i][j][k]*mhd->bz[i][j][k]);
				vexbz[i][j][k]=mhd->zd[i][j][k]*mhd->e*mdinv*mhd->rho[i][j][k]*(vex[i][j][k]*mhd->by[i][j][k]-vey[i][j][k]*mhd->bx[i][j][k]);
				//	ivixb term
				ivixbx[i][j][k]=mhd->zi[i][j][k]*mhd->e*miinv*(mhd->siy[i][j][k]*mhd->bz[i][j][k]-mhd->siz[i][j][k]*mhd->by[i][j][k]);
				ivixby[i][j][k]=mhd->zi[i][j][k]*mhd->e*miinv*(mhd->siz[i][j][k]*mhd->bx[i][j][k]-mhd->six[i][j][k]*mhd->bz[i][j][k]);
				ivixbz[i][j][k]=mhd->zi[i][j][k]*mhd->e*miinv*(mhd->six[i][j][k]*mhd->by[i][j][k]-mhd->siy[i][j][k]*mhd->bx[i][j][k]);
				//	ivexb term
				ivexbx[i][j][k]=mhd->zi[i][j][k]*mhd->e*miinv*mhd->rhoi[i][j][k]*(vey[i][j][k]*mhd->bz[i][j][k]-vez[i][j][k]*mhd->by[i][j][k]);
				ivexby[i][j][k]=mhd->zi[i][j][k]*mhd->e*miinv*mhd->rhoi[i][j][k]*(vez[i][j][k]*mhd->bx[i][j][k]-vex[i][j][k]*mhd->bz[i][j][k]);
				ivexbz[i][j][k]=mhd->zi[i][j][k]*mhd->e*miinv*mhd->rhoi[i][j][k]*(vex[i][j][k]*mhd->by[i][j][k]-vey[i][j][k]*mhd->bx[i][j][k]);
				//     Collisional & Ionization Source Terms
				qci[i][j][k]=mimime*(mhd->ioniz*mhd->rhon[i][j][k]-mhd->recom*mhd->rhoi[i][j][k]*mhd->rhoe[i][j][k]*meinv);
				qcn[i][j][k]=-1.*mhd->ioniz*mhd->rhon[i][j][k]+mhd->recom*mhd->rhoi[i][j][k]*mhd->rhoe[i][j][k]*meinv;
				qsdx[i][j][k]=-1.*mhd->rhoi[i][j][k]*mhd->nuid[i][j][k]*mdpmi*(vx[i][j][k]-vix[i][j][k])
						 -mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*(vx[i][j][k]-vex[i][j][k])
						 -mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*(vx[i][j][k]-vnx[i][j][k]);
				qsdy[i][j][k]=-1.*mhd->rhoi[i][j][k]*mhd->nuid[i][j][k]*mdpmi*(vy[i][j][k]-viy[i][j][k])
						 -mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*(vy[i][j][k]-vey[i][j][k])
						 -mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*(vy[i][j][k]-vny[i][j][k]);
				qsdz[i][j][k]=-1.*mhd->rhoi[i][j][k]*mhd->nuid[i][j][k]*mdpmi*(vz[i][j][k]-viz[i][j][k])
						 -mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*(vz[i][j][k]-vez[i][j][k])
						 -mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*(vz[i][j][k]-vnz[i][j][k]);
				qsix[i][j][k]=-1.*mhd->rhoi[i][j][k]*mhd->nuid[i][j][k]*mdpmi*(vix[i][j][k]-vx[i][j][k])
						 -mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*mipme*(vix[i][j][k]-vex[i][j][k])
						 -mhd->rhoi[i][j][k]*mhd->nuin[i][j][k]*mipmn*(vix[i][j][k]-vnx[i][j][k])
						 +mhd->mi*mipme*mhd->ioniz*mhd->snx[i][j][k]
						 -mimime*mhd->recom*mhd->six[i][j][k]*mhd->rhoe[i][j][k]*meinv;
				qsiy[i][j][k]=-1.*mhd->rhoi[i][j][k]*mhd->nuid[i][j][k]*mdpmi*(viy[i][j][k]-vy[i][j][k])
						 -mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*mipme*(viy[i][j][k]-vey[i][j][k])
						 -mhd->rhoi[i][j][k]*mhd->nuin[i][j][k]*mipmn*(viy[i][j][k]-vny[i][j][k])
						 +mhd->mi*mipme*mhd->ioniz*mhd->sny[i][j][k]
						 -mimime*mhd->recom*mhd->siy[i][j][k]*mhd->rhoe[i][j][k]*meinv;
				qsiz[i][j][k]=-1.*mhd->rhoi[i][j][k]*mhd->nuid[i][j][k]*mdpmi*(viz[i][j][k]-vz[i][j][k])
						 -mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*mipme*(viz[i][j][k]-vez[i][j][k])
						 -mhd->rhoi[i][j][k]*mhd->nuin[i][j][k]*mipmn*(viz[i][j][k]-vnz[i][j][k])
						 +mhd->mi*mipme*mhd->ioniz*mhd->snz[i][j][k]
						 -mimime*mhd->recom*mhd->siz[i][j][k]*mhd->rhoe[i][j][k]*meinv;
				qsex[i][j][k]=-1.*mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*(vex[i][j][k]-vx[i][j][k])
						 -mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*mipme*(vex[i][j][k]-vix[i][j][k])
						 -mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*(vex[i][j][k]-vnx[i][j][k])
						 +mhd->me*mipme*mhd->ioniz*mhd->snx[i][j][k]
						 -mipme*mhd->recom*mhd->rhoi[i][j][k]*mhd->sex[i][j][k];
				qsey[i][j][k]=-1.*mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*(vey[i][j][k]-vy[i][j][k])
						 -mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*mipme*(vey[i][j][k]-viy[i][j][k])
						 -mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*(vey[i][j][k]-vny[i][j][k])
						 +mhd->me*mipme*mhd->ioniz*mhd->sny[i][j][k]
						 -mipme*mhd->recom*mhd->rhoi[i][j][k]*mhd->sey[i][j][k];
				qsez[i][j][k]=-1.*mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*(vez[i][j][k]-vz[i][j][k])
						 -mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*mipme*(vez[i][j][k]-viz[i][j][k])
						 -mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*(vez[i][j][k]-vnz[i][j][k])
						 +mhd->me*mipme*mhd->ioniz*mhd->snz[i][j][k]
						 -mipme*mhd->recom*mhd->rhoi[i][j][k]*mhd->sez[i][j][k];
				qsnx[i][j][k]=-1.*mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*(vnx[i][j][k]-vx[i][j][k])
						 -mhd->rhoi[i][j][k]*mhd->nuin[i][j][k]*mipmn*(vnx[i][j][k]-vix[i][j][k])
						 -mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*(vnx[i][j][k]-vex[i][j][k])
						 -mhd->ioniz*mhd->snx[i][j][k]
						 +mhd->recom*mipme*(mhd->six[i][j][k]*mhd->rhoe[i][j][k]/memi+mhd->sex[i][j][k]*mhd->rhoi[i][j][k]);
				qsny[i][j][k]=-1.*mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*(vny[i][j][k]-vy[i][j][k])
						 -mhd->rhoi[i][j][k]*mhd->nuin[i][j][k]*mipmn*(vny[i][j][k]-viy[i][j][k])
						 -mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*(vny[i][j][k]-vey[i][j][k])
						 -mhd->ioniz*mhd->sny[i][j][k]
						 +mhd->recom*mipme*(mhd->siy[i][j][k]*mhd->rhoe[i][j][k]/memi+mhd->sey[i][j][k]*mhd->rhoi[i][j][k]);
				qsnz[i][j][k]=-1.*mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*(vnz[i][j][k]-vz[i][j][k])
						 -mhd->rhoi[i][j][k]*mhd->nuin[i][j][k]*mipmn*(vnz[i][j][k]-viz[i][j][k])
						 -mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*(vnz[i][j][k]-vez[i][j][k])
						 -mhd->ioniz*mhd->snz[i][j][k]
						 +mhd->recom*mipme*(mhd->siz[i][j][k]*mhd->rhoe[i][j][k]/memi+mhd->sez[i][j][k]*mhd->rhoi[i][j][k]);
				qed[i][j][k]=mhd->rhoi[i][j][k]*mhd->nuid[i][j][k]*mdpmi*( (vix[i][j][k]-vx[i][j][k])*(vix[i][j][k]-vx[i][j][k])
											  +(viy[i][j][k]-vy[i][j][k])*(viy[i][j][k]-vy[i][j][k])
											  +(viz[i][j][k]-vz[i][j][k])*(viz[i][j][k]-vz[i][j][k]))
					    +mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*( (vex[i][j][k]-vx[i][j][k])*(vex[i][j][k]-vx[i][j][k])
											 +(vey[i][j][k]-vy[i][j][k])*(vey[i][j][k]-vy[i][j][k])
											 +(vez[i][j][k]-vz[i][j][k])*(vez[i][j][k]-vz[i][j][k]))
					    +mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*( (vnx[i][j][k]-vx[i][j][k])*(vnx[i][j][k]-vx[i][j][k])
											 +(vny[i][j][k]-vy[i][j][k])*(vny[i][j][k]-vy[i][j][k])
											 +(vnz[i][j][k]-vz[i][j][k])*(vnz[i][j][k]-vz[i][j][k]))
					    -2.*mhd->rhoi[i][j][k]*mhd->nuid[i][j][k]*mdpmi*(mhd->p[i][j][k]*gaminv[i][j][k]*rhoinv[i][j][k]-mhd->mi*mhd->pi[i][j][k]*gamiinv[i][j][k]*rhoiinv[i][j][k])
					    -2.*mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*(mhd->p[i][j][k]*gaminv[i][j][k]*rhoinv[i][j][k]-mhd->me*mhd->pe[i][j][k]*gameinv[i][j][k]*rhoeinv[i][j][k])
					    -2.*mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*(mhd->p[i][j][k]*gaminv[i][j][k]*rhoinv[i][j][k]-mhd->mn*mhd->pn[i][j][k]*gamninv[i][j][k]*rhoninv[i][j][k]);
				qei[i][j][k]=mhd->rhoi[i][j][k]*( mhd->nuid[i][j][k]*mdpmi*( (vx[i][j][k]-vix[i][j][k])*(vx[i][j][k]-vix[i][j][k])
											   +(vy[i][j][k]-viy[i][j][k])*(vy[i][j][k]-viy[i][j][k])
											   +(vz[i][j][k]-viz[i][j][k])*(vz[i][j][k]-viz[i][j][k]))
								 +mhd->nuie[i][j][k]*mipme*( (vex[i][j][k]-vix[i][j][k])*(vex[i][j][k]-vix[i][j][k])
											    +(vey[i][j][k]-viy[i][j][k])*(vey[i][j][k]-viy[i][j][k])
											    +(vez[i][j][k]-viz[i][j][k])*(vez[i][j][k]-viz[i][j][k]))
								 +mhd->nuin[i][j][k]*mipmn*( (vnx[i][j][k]-vix[i][j][k])*(vnx[i][j][k]-vix[i][j][k])
											    +(vny[i][j][k]-viy[i][j][k])*(vny[i][j][k]-viy[i][j][k])
											    +(vnz[i][j][k]-viz[i][j][k])*(vnz[i][j][k]-viz[i][j][k])))
					    -2.*mhd->rhoi[i][j][k]*mhd->nuid[i][j][k]*mdpmi*(mhd->mi*mhd->pi[i][j][k]*gamiinv[i][j][k]*rhoiinv[i][j][k]-mhd->p[i][j][k]*gaminv[i][j][k]*rhoinv[i][j][k])
					    -2.*mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*mipme*(mhd->mi*mhd->pi[i][j][k]*gamiinv[i][j][k]*rhoiinv[i][j][k]-mhd->me*mhd->pe[i][j][k]*gameinv[i][j][k]*rhoeinv[i][j][k])
					    -2.*mhd->rhoi[i][j][k]*mhd->nuin[i][j][k]*mipmn*(mhd->mi*mhd->pi[i][j][k]*gamiinv[i][j][k]*rhoiinv[i][j][k]-mhd->mn*mhd->pn[i][j][k]*gamninv[i][j][k]*rhoninv[i][j][k])
					    +mhd->gamman[i][j][k]*mipme*mhd->pn[i][j][k]*mhd->ioniz*mhd->mn*gamninv[i][j][k]
					    -mhd->gammai[i][j][k]*mhd->pi[i][j][k]*mhd->recom*mhd->rhoe[i][j][k]*meinv*gamiinv[i][j][k]
					    +0.25*mhd->mi*mipme*( mhd->ioniz*mhd->rhon[i][j][k]
								 +mhd->recom*mhd->rhoi[i][j][k]*mhd->rhoe[i][j][k]*meinv)*( (vnx[i][j][k]-vix[i][j][k])*(vnx[i][j][k]-vix[i][j][k])
															   +(vny[i][j][k]-viy[i][j][k])*(vny[i][j][k]-viy[i][j][k])
															   +(vnz[i][j][k]-viz[i][j][k])*(vnz[i][j][k]-viz[i][j][k]));
				qee[i][j][k]= mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*( (vx[i][j][k]-vex[i][j][k])*(vx[i][j][k]-vex[i][j][k])
											  +(vy[i][j][k]-vey[i][j][k])*(vy[i][j][k]-vey[i][j][k])
											  +(vz[i][j][k]-vez[i][j][k])*(vz[i][j][k]-vez[i][j][k]))
					    +mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*mipme*( (vix[i][j][k]-vex[i][j][k])*(vix[i][j][k]-vex[i][j][k])
											  +(viy[i][j][k]-vey[i][j][k])*(viy[i][j][k]-vey[i][j][k])
											  +(viz[i][j][k]-vez[i][j][k])*(viz[i][j][k]-vez[i][j][k]))
					    +mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*( (vnx[i][j][k]-vex[i][j][k])*(vnx[i][j][k]-vex[i][j][k])
											  +(vny[i][j][k]-vey[i][j][k])*(vny[i][j][k]-vey[i][j][k])
											  +(vnz[i][j][k]-vez[i][j][k])*(vnz[i][j][k]-vez[i][j][k]))
					    -2.*mhd->rho[i][j][k]*mhd->nude[i][j][k]*mdpme*(mhd->me*mhd->pe[i][j][k]*gameinv[i][j][k]*rhoeinv[i][j][k]-mhd->p[i][j][k]*gaminv[i][j][k]*rhoinv[i][j][k])
					    -2.*mhd->rhoi[i][j][k]*mhd->nuie[i][j][k]*mipme*(mhd->me*mhd->pe[i][j][k]*gameinv[i][j][k]*rhoeinv[i][j][k]-mhd->mi*mhd->pi[i][j][k]*gamiinv[i][j][k]*rhoiinv[i][j][k])
					    -2.*mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*(mhd->me*mhd->pe[i][j][k]*gameinv[i][j][k]*rhoeinv[i][j][k]-mhd->mn*mhd->pn[i][j][k]*gamninv[i][j][k]*rhoninv[i][j][k])
					    +mhd->gamman[i][j][k]*mipme*mhd->pn[i][j][k]*mhd->ioniz*memi*mhd->mn*gamninv[i][j][k]
					    -mhd->gammae[i][j][k]*mhd->pe[i][j][k]*mhd->recom*mhd->rhoi[i][j][k]*miinv*gameinv[i][j][k]
					    +0.25*mipme*( mhd->ioniz*mhd->rhon[i][j][k]*mhd->me
							 +mhd->recom*mhd->rhoi[i][j][k]*mhd->rhoe[i][j][k])*( (vnx[i][j][k]-vex[i][j][k])*(vnx[i][j][k]-vex[i][j][k])
													     +(vny[i][j][k]-vey[i][j][k])*(vny[i][j][k]-vey[i][j][k])
													     +(vnz[i][j][k]-vez[i][j][k])*(vnz[i][j][k]-vez[i][j][k]));
				qen[i][j][k]= mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*(  (vnx[i][j][k]-vx[i][j][k])*(vnx[i][j][k]-vx[i][j][k])
											   +(vny[i][j][k]-vy[i][j][k])*(vny[i][j][k]-vy[i][j][k])
											   +(vnz[i][j][k]-vz[i][j][k])*(vnz[i][j][k]-vz[i][j][k]))
					     +mhd->rhoi[i][j][k]*mhd->nuin[i][j][k]*mipmn*( (vnx[i][j][k]-vix[i][j][k])*(vnx[i][j][k]-vix[i][j][k])
											   +(vny[i][j][k]-viy[i][j][k])*(vny[i][j][k]-viy[i][j][k])
											   +(vnz[i][j][k]-viz[i][j][k])*(vnz[i][j][k]-viz[i][j][k]))
					     +mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*( (vnx[i][j][k]-vex[i][j][k])*(vnx[i][j][k]-vex[i][j][k])
											   +(vny[i][j][k]-vey[i][j][k])*(vny[i][j][k]-vey[i][j][k])
											   +(vnz[i][j][k]-vez[i][j][k])*(vnz[i][j][k]-vez[i][j][k]))
					     -2.*mhd->rho[i][j][k]*mhd->nudn[i][j][k]*mdpmn*(mhd->mn*mhd->pn[i][j][k]*rhoninv[i][j][k]*gamninv[i][j][k]-mhd->p[i][j][k]*rhoinv[i][j][k]*gaminv[i][j][k])
					     -2.*mhd->rhoi[i][j][k]*mhd->nuin[i][j][k]*mipmn*(mhd->mn*mhd->pn[i][j][k]*rhoninv[i][j][k]*gamninv[i][j][k]-mhd->mi*mhd->pi[i][j][k]*rhoiinv[i][j][k]*gamiinv[i][j][k])
					     -2.*mhd->rhoe[i][j][k]*mhd->nuen[i][j][k]*mepmn*(mhd->mn*mhd->pn[i][j][k]*rhoninv[i][j][k]*gamninv[i][j][k]-mhd->me*mhd->pe[i][j][k]*rhoeinv[i][j][k]*gameinv[i][j][k])
					     -mhd->gamman[i][j][k]*mhd->pn[i][j][k]*mhd->ioniz*gamninv[i][j][k]
					     +( mhd->gammae[i][j][k]*mhd->pe[i][j][k]*rhoeinv[i][j][k]*mhd->me*gameinv[i][j][k]
					      +mhd->gammai[i][j][k]*mhd->mi*mhd->pi[i][j][k]*rhoiinv[i][j][k]*gamiinv[i][j][k])*mhd->recom*mhd->rhoi[i][j][k]*mhd->rhoe[i][j][k]*miinv*meinv
					     +0.25*mhd->mi*mipme*( mhd->ioniz*mhd->rhon[i][j][k]
								  +mhd->recom*mhd->rhoi[i][j][k]*mhd->rhoe[i][j][k]*meinv)*( (vnx[i][j][k]-vix[i][j][k])*(vnx[i][j][k]-vix[i][j][k])
															    +(vny[i][j][k]-viy[i][j][k])*(vny[i][j][k]-viy[i][j][k])
															    +(vnz[i][j][k]-viz[i][j][k])*(vnz[i][j][k]-viz[i][j][k]))
					     +0.25*mipme*( mhd->ioniz*mhd->rhon[i][j][k]*mhd->me
							  +mhd->recom*mhd->rhoi[i][j][k]*mhd->rhoe[i][j][k])*( (vnx[i][j][k]-vex[i][j][k])*(vnx[i][j][k]-vex[i][j][k])
													      +(vny[i][j][k]-vey[i][j][k])*(vny[i][j][k]-vey[i][j][k])
													      +(vnz[i][j][k]-vez[i][j][k])*(vnz[i][j][k]-vez[i][j][k]));
				}
			}
		}
		// Because of rhoeinv reference we need to place these in a separate loop
		#pragma omp for
		for(i=1;i<grid->nxm1;i++)
		{
			for(j=1;j<grid->nym1;j++)
			{
				for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
				{
				cdsx[i][j][k]=-grid->difx[i][j][k]*memd*mhd->zd[i][j][k]*mhd->rho[i][j][k]*rhoeinv[i][j][k]*(mhd->pe[i+1][j][k]-mhd->pe[i-1][j][k]);
				cdsy[i][j][k]=-grid->dify[i][j][k]*memd*mhd->zd[i][j][k]*mhd->rho[i][j][k]*rhoeinv[i][j][k]*(mhd->pe[i][j+1][k]-mhd->pe[i][j-1][k]);
				cdsz[i][j][k]=-grid->difz[i][j][k]*memd*mhd->zd[i][j][k]*mhd->rho[i][j][k]*rhoeinv[i][j][k]*(mhd->pe[i][j][k+1]-mhd->pe[i][j][k-1]);
				cdsix[i][j][k]=grid->difx[i][j][k]*memd*mhd->zi[i][j][k]*mhd->rhoi[i][j][k]*rhoeinv[i][j][k]*(mhd->pe[i+1][j][k]-mhd->pe[i-1][j][k]);
				cdsiy[i][j][k]=grid->dify[i][j][k]*memd*mhd->zi[i][j][k]*mhd->rhoi[i][j][k]*rhoeinv[i][j][k]*(mhd->pe[i][j+1][k]-mhd->pe[i][j-1][k]);
				cdsiz[i][j][k]=grid->difz[i][j][k]*memd*mhd->zi[i][j][k]*mhd->rhoi[i][j][k]*rhoeinv[i][j][k]*(mhd->pe[i][j][k+1]-mhd->pe[i][j][k-1]);
				cdpd[i][j][k]=+(mhd->gamma[i][j][k]-1.)*mhd->p[i][j][k]*( grid->difx[i][j][k]*(vx[i+1][j][k]-vx[i-1][j][k])
											   +grid->dify[i][j][k]*(vy[i][j+1][k]-vy[i][j-1][k])
											   +grid->difz[i][j][k]*(vz[i][j][k+1]-vz[i][j][k-1]));
				cdpi[i][j][k]=+(mhd->gammai[i][j][k]-1.)*mhd->pi[i][j][k]*( grid->difx[i][j][k]*(vix[i+1][j][k]-vix[i-1][j][k])
											   +grid->dify[i][j][k]*(viy[i][j+1][k]-viy[i][j-1][k])
											   +grid->difz[i][j][k]*(viz[i][j][k+1]-viz[i][j][k-1]));
				cdpe[i][j][k]=+(mhd->gammae[i][j][k]-1.)*mhd->pe[i][j][k]*( grid->difx[i][j][k]*(vex[i+1][j][k]-vex[i-1][j][k])
											   +grid->dify[i][j][k]*(vey[i][j+1][k]-vey[i][j-1][k])
											   +grid->difz[i][j][k]*(vez[i][j][k+1]-vez[i][j][k-1]));
				cdpn[i][j][k]=+(mhd->gamman[i][j][k]-1.)*mhd->pn[i][j][k]*( grid->difx[i][j][k]*(vnx[i+1][j][k]-vnx[i-1][j][k])
											   +grid->dify[i][j][k]*(vny[i][j+1][k]-vny[i][j-1][k])
											   +grid->difz[i][j][k]*(vnz[i][j][k+1]-vnz[i][j][k-1]));
				}
			}
		}
		//Second step g(t,i)=u(t,i)-(alpha*dt/2)*(u(t,i+1)-u(t,i+1))+alpha*dt*u(t,i)
//		if (tid == 0)
//		{
//			printf("...............%04.2f [s] -----\n",(omp_get_wtime()-stime)/TIMESCALE);
//			stime=omp_get_wtime();
//			printf("----- Flux Correcting Step");
//		}
//		#pragma omp barrier
		#pragma omp for
		for(i=1;i<grid->nxm1;i++)
		{
			for(j=1;j<grid->nym1;j++)
			{
				for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
				{
					//Densities (no dust source term)
					mhd->rhoi[i][j][k]=mhd->rhoi[i][j][k]-grid->dt*spat*avgqci[i][j][k]
									     +grid->dt*qci[i][j][k];
					mhd->rhon[i][j][k]=mhd->rhon[i][j][k]-grid->dt*spat*avgqcn[i][j][k]
									     +grid->dt*qcn[i][j][k];
					//Dust Momentum
					mhd->sx[i][j][k]=mhd->sx[i][j][k]+grid->dt*spat*avgqsx[i][j][k]
									 -grid->dt*( vxbx[i][j][k]
										    -vexbx[i][j][k]
										    -srho[i][j][k]*mhd->gravx[i][j][k]
										    -qsdx[i][j][k]
										    +memd*srho[i][j][k]*mhd->zd[i][j][k]*rhoeinv[i][j][k]*(srhoe[i][j][k]*mhd->gravx[i][j][k]+qsex[i][j][k]))
					                               +grid->dt*avgdsx[i][j][k]-grid->dt*cdsx[i][j][k];
					mhd->sy[i][j][k]=mhd->sy[i][j][k]+grid->dt*spat*avgqsy[i][j][k]
									 -grid->dt*( vxby[i][j][k]
										    -vexby[i][j][k]
										    -srho[i][j][k]*mhd->gravy[i][j][k]
										    -qsdy[i][j][k]
										    +memd*srho[i][j][k]*mhd->zd[i][j][k]*rhoeinv[i][j][k]*(srhoe[i][j][k]*mhd->gravy[i][j][k]+qsey[i][j][k]))
					                               +grid->dt*avgdsy[i][j][k]-grid->dt*cdsy[i][j][k];
					mhd->sz[i][j][k]=mhd->sz[i][j][k]+grid->dt*spat*avgqsz[i][j][k]
									 -grid->dt*( vxbz[i][j][k]
										    -vexbz[i][j][k]
										    -srho[i][j][k]*mhd->gravz[i][j][k]
										    -qsdz[i][j][k]
										    +memd*srho[i][j][k]*mhd->zd[i][j][k]*rhoeinv[i][j][k]*(srhoe[i][j][k]*mhd->gravz[i][j][k]+qsez[i][j][k]))
					                               +grid->dt*avgdsz[i][j][k]-grid->dt*cdsz[i][j][k];
					//Ion Momentum
					mhd->six[i][j][k]=mhd->six[i][j][k]-grid->dt*spat*avgqsix[i][j][k]
									   +grid->dt*( ivixbx[i][j][k]
										      -ivexbx[i][j][k]
										      +srhoi[i][j][k]*mhd->gravx[i][j][k]
										      +qsix[i][j][k]
										      +memi*srhoi[i][j][k]*mhd->zi[i][j][k]*rhoeinv[i][j][k]*(srhoe[i][j][k]*mhd->gravx[i][j][k]+qsex[i][j][k]))
					                               +grid->dt*avgdsix[i][j][k]-grid->dt*cdsix[i][j][k];
					mhd->siy[i][j][k]=mhd->siy[i][j][k]-grid->dt*spat*avgqsiy[i][j][k]
									   +grid->dt*( ivixby[i][j][k]
										      -ivexby[i][j][k]
										      +srhoi[i][j][k]*mhd->gravy[i][j][k]
										      +qsiy[i][j][k]
										      +memi*srhoi[i][j][k]*mhd->zi[i][j][k]*rhoeinv[i][j][k]*(srhoe[i][j][k]*mhd->gravy[i][j][k]+qsey[i][j][k]))
					                                   +grid->dt*avgdsiy[i][j][k]-grid->dt*cdsiy[i][j][k];
					mhd->siz[i][j][k]=mhd->siz[i][j][k]-grid->dt*spat*avgqsiz[i][j][k]
									   +grid->dt*( ivixbz[i][j][k]
										      -ivexbz[i][j][k]
										      +srhoi[i][j][k]*mhd->gravz[i][j][k]
										      +qsiz[i][j][k]
										      +memi*srhoi[i][j][k]*mhd->zi[i][j][k]*rhoeinv[i][j][k]*(srhoe[i][j][k]*mhd->gravz[i][j][k]+qsez[i][j][k]))
					                                   +grid->dt*avgdsiz[i][j][k]-grid->dt*cdsiz[i][j][k];
					//Neutral Momentum
					mhd->snx[i][j][k]=mhd->snx[i][j][k]+spat*grid->dt*avgqsnx[i][j][k]-grid->dt*(-1.*srhon[i][j][k]*mhd->gravx[i][j][k]-qsnx[i][j][k]);
					mhd->sny[i][j][k]=mhd->sny[i][j][k]+spat*grid->dt*avgqsny[i][j][k]-grid->dt*(-1.*srhon[i][j][k]*mhd->gravy[i][j][k]-qsny[i][j][k]);
					mhd->snz[i][j][k]=mhd->snz[i][j][k]+spat*grid->dt*avgqsnz[i][j][k]-grid->dt*(-1.*srhon[i][j][k]*mhd->gravz[i][j][k]-qsnz[i][j][k]);
					//Pressure
					mhd->p[i][j][k]=mhd->p[i][j][k]-grid->dt*spat*(mhd->gamma[i][j][k]-1.)*avgqed[i][j][k]
								       +grid->dt*(mhd->gamma[i][j][k]-1.)*qed[i][j][k]
					                               +grid->dt*avgdpd[i][j][k]-grid->dt*cdpd[i][j][k];
					mhd->pi[i][j][k]=mhd->pi[i][j][k]-grid->dt*spat*(mhd->gammai[i][j][k]-1.)*avgqei[i][j][k]
									 +grid->dt*(mhd->gammai[i][j][k]-1.)*qei[i][j][k]
					                                 +grid->dt*avgdpi[i][j][k]-grid->dt*cdpi[i][j][k];
					mhd->pe[i][j][k]=mhd->pe[i][j][k]-grid->dt*spat*(mhd->gammae[i][j][k]-1.)*avgqee[i][j][k]
									 +grid->dt*(mhd->gammae[i][j][k]-1.)*qee[i][j][k]
					                                 +grid->dt*avgdpe[i][j][k]-grid->dt*cdpe[i][j][k];
					mhd->pn[i][j][k]=mhd->pn[i][j][k]-grid->dt*spat*(mhd->gamman[i][j][k]-1.)*avgqen[i][j][k]
									 +grid->dt*(mhd->gamman[i][j][k]-1.)*qen[i][j][k]
					                                 +grid->dt*avgdpn[i][j][k]-grid->dt*cdpn[i][j][k];
					//Magnetic Field (no source terms)
				}
			}
		}
		//Update Electron Continuity (removed from previous for loop for parallelization due to rho reference.
		#pragma omp for
		for(i=1;i<grid->nxm1;i++)
		{
			for(j=1;j<grid->nym1;j++)
			{
				for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
				{
					mhd->rhoe[i][j][k]= (1-grid->econt)*mhd->me*(mhd->zi[i][j][k]*mhd->rhoi[i][j][k]*miinv-mhd->zd[i][j][k]*mhd->rho[i][j][k]*mdinv)
					                   +grid->econt*(mhd->rhoe[i][j][k]);
				}
			}
		}
//		if (tid == 0)
//		{
//			printf("..............%04.2f [s] -----\n",(omp_get_wtime()-stime)/TIMESCALE);
//			stime=omp_get_wtime();
//			printf("----- Check for smoothing");
//		}
//		#pragma omp barrier
		//Grids are now at the same timestep so we smooth
		#pragma omp single nowait
		smooth(mhd->rho,mhd->rhosmo,grid,tid);
		#pragma omp single nowait
		smooth(mhd->rhoi,mhd->rhoismo,grid,tid);
		#pragma omp single nowait
		smooth(mhd->rhon,mhd->rhonsmo,grid,tid);
		#pragma omp single nowait
		smooth(mhd->sx,mhd->sxsmo,grid,tid);
		#pragma omp single nowait
		smooth(mhd->sy,mhd->sysmo,grid,tid);
		#pragma omp single nowait
		smooth(mhd->sz,mhd->szsmo,grid,tid);
		#pragma omp single nowait
		smooth(mhd->six,mhd->sixsmo,grid,tid);
		#pragma omp single nowait
		smooth(mhd->siy,mhd->siysmo,grid,tid);
		#pragma omp single nowait
		smooth(mhd->siz,mhd->sizsmo,grid,tid);
		#pragma omp single nowait
		smooth(mhd->snx,mhd->snxsmo,grid,tid);
		#pragma omp single nowait
		smooth(mhd->sny,mhd->snysmo,grid,tid);
		#pragma omp single nowait
		smooth(mhd->snz,mhd->snzsmo,grid,tid);
		#pragma omp single nowait
		smooth(mhd->p,mhd->psmo,grid,tid);
		#pragma omp single nowait
		smooth(mhd->pi,mhd->pismo,grid,tid);
		#pragma omp single nowait
		smooth(mhd->pe,mhd->pesmo,grid,tid);
		#pragma omp single nowait
		smooth(mhd->pn,mhd->pnsmo,grid,tid);
//		#pragma omp barrier
//		if (tid == 0)
//		{
//			printf("...............%04.2f [s] -----\n",(omp_get_wtime()-stime)/TIMESCALE);
//			stime=omp_get_wtime();
//			printf("----- Smooth");
//		}
//		#pragma omp barrier
		#pragma omp sections nowait
		{
			#pragma omp section
			{
				for(i=1;i<grid->nxm1;i++)
				{
					for(j=1;j<grid->nym1;j++)
					{
						for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
						{
							if (mhd->rhosmo[i][j][k] == 1)
							{
								mhd->rho[i][j][k]=grid->s[i][j][k]*mhd->rho[i][j][k]+grid->ssx[i][j][k]*(mhd->rho[i+1][j][k]+mhd->rho[i-1][j][k])
								                                                    +grid->ssy[i][j][k]*(mhd->rho[i][j+1][k]+mhd->rho[i][j-1][k])
								                                                    +grid->ssz[i][j][k]*(mhd->rho[i][j][k+1]+mhd->rho[i][j][k-1]);
								//Old Way (not correct)
								//drho[i][j][k]=drho[i][j][k]-( mhd->visx*(grid->meanpx[i][j][k]*mhd->rho[i+1][j][k]+grid->meanmx[i][j][k]*mhd->rho[i-1][j][k])
								//				     +mhd->visy*(grid->meanpy[i][j][k]*mhd->rho[i][j+1][k]+grid->meanmy[i][j][k]*mhd->rho[i][j-1][k])
								//				     +mhd->visz*(grid->meanpz[i][j][k]*mhd->rho[i][j][k+1]+grid->meanmz[i][j][k]*mhd->rho[i][j][k-1])
								//				     -(mhd->visx+mhd->visy+mhd->visz)*mhd->rho[i][j][k])*dtinv;
							}
						}
					}
				}
			}
			#pragma omp section
			{
				for(i=1;i<grid->nxm1;i++)
				{
					for(j=1;j<grid->nym1;j++)
					{
						for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
						{
							if (mhd->rhoismo[i][j][k] == 1)
							{
								mhd->rhoi[i][j][k]=grid->s[i][j][k]*mhd->rhoi[i][j][k]+grid->ssx[i][j][k]*(mhd->rhoi[i+1][j][k]+mhd->rhoi[i-1][j][k])
								                                                      +grid->ssy[i][j][k]*(mhd->rhoi[i][j+1][k]+mhd->rhoi[i][j-1][k])
								                                                      +grid->ssz[i][j][k]*(mhd->rhoi[i][j][k+1]+mhd->rhoi[i][j][k-1]);
							}
						}
					}
				}
			}
			#pragma omp section
			{
				for(i=1;i<grid->nxm1;i++)
				{
					for(j=1;j<grid->nym1;j++)
					{
						for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
						{
							if (mhd->rhonsmo[i][j][k] == 1)
							{
								mhd->rhon[i][j][k]=grid->s[i][j][k]*mhd->rhon[i][j][k]+grid->ssx[i][j][k]*(mhd->rhon[i+1][j][k]+mhd->rhon[i-1][j][k])
								                                                      +grid->ssy[i][j][k]*(mhd->rhon[i][j+1][k]+mhd->rhon[i][j-1][k])
								                                                      +grid->ssz[i][j][k]*(mhd->rhon[i][j][k+1]+mhd->rhon[i][j][k-1]);
							}
						}
					}
				}
			}
			#pragma omp section
			{
				for(i=1;i<grid->nxm1;i++)
				{
					for(j=1;j<grid->nym1;j++)
					{
						for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
						{
							if (mhd->psmo[i][j][k] == 1)
							{
								mhd->p[i][j][k]=grid->s[i][j][k]*mhd->p[i][j][k]+grid->ssx[i][j][k]*(mhd->p[i+1][j][k]+mhd->p[i-1][j][k])
								                                                +grid->ssy[i][j][k]*(mhd->p[i][j+1][k]+mhd->p[i][j-1][k])
								                                                +grid->ssz[i][j][k]*(mhd->p[i][j][k+1]+mhd->p[i][j][k-1]);
							}
						}
					}
				}
			}
			#pragma omp section
			{
				for(i=1;i<grid->nxm1;i++)
				{
					for(j=1;j<grid->nym1;j++)
					{
						for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
						{
							if (mhd->pismo[i][j][k] == 1)
							{
								mhd->pi[i][j][k]=grid->s[i][j][k]*mhd->pi[i][j][k]+grid->ssx[i][j][k]*(mhd->pi[i+1][j][k]+mhd->pi[i-1][j][k])
								                                                  +grid->ssy[i][j][k]*(mhd->pi[i][j+1][k]+mhd->pi[i][j-1][k])
								                                                  +grid->ssz[i][j][k]*(mhd->pi[i][j][k+1]+mhd->pi[i][j][k-1]);
							}
						}
					}
				}
			}
			#pragma omp section
			{
				for(i=1;i<grid->nxm1;i++)
				{
					for(j=1;j<grid->nym1;j++)
					{
						for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
						{
							if (mhd->pesmo[i][j][k] == 1)
							{
								mhd->pe[i][j][k]=grid->s[i][j][k]*mhd->pe[i][j][k]+grid->ssx[i][j][k]*(mhd->pe[i+1][j][k]+mhd->pe[i-1][j][k])
								                                                  +grid->ssy[i][j][k]*(mhd->pe[i][j+1][k]+mhd->pe[i][j-1][k])
								                                                  +grid->ssz[i][j][k]*(mhd->pe[i][j][k+1]+mhd->pe[i][j][k-1]);
							}
						}
					}
				}
			}
			#pragma omp section
			{
				for(i=1;i<grid->nxm1;i++)
				{
					for(j=1;j<grid->nym1;j++)
					{
						for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
						{
							if (mhd->pnsmo[i][j][k] == 1)
							{
								mhd->pn[i][j][k]=grid->s[i][j][k]*mhd->pn[i][j][k]+grid->ssx[i][j][k]*(mhd->pn[i+1][j][k]+mhd->pn[i-1][j][k])
								                                                  +grid->ssy[i][j][k]*(mhd->pn[i][j+1][k]+mhd->pn[i][j-1][k])
								                                                  +grid->ssz[i][j][k]*(mhd->pn[i][j][k+1]+mhd->pn[i][j][k-1]);
							}
						}
					}
				}
			}
			#pragma omp section
			{
				for(i=1;i<grid->nxm1;i++)
				{
					for(j=1;j<grid->nym1;j++)
					{
						for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
						{
							if (mhd->sxsmo[i][j][k] == 1)
							{
								mhd->sx[i][j][k]=grid->s[i][j][k]*mhd->sx[i][j][k]+grid->ssx[i][j][k]*(mhd->sx[i+1][j][k]+mhd->sx[i-1][j][k])
								                                                  +grid->ssy[i][j][k]*(mhd->sx[i][j+1][k]+mhd->sx[i][j-1][k])
								                                                  +grid->ssz[i][j][k]*(mhd->sx[i][j][k+1]+mhd->sx[i][j][k-1]);
							}
						}
					}
				}
			}
			#pragma omp section
			{
				for(i=1;i<grid->nxm1;i++)
				{
					for(j=1;j<grid->nym1;j++)
					{
						for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
						{
							if (mhd->sysmo[i][j][k] == 1)
							{
								mhd->sy[i][j][k]=grid->s[i][j][k]*mhd->sy[i][j][k]+grid->ssx[i][j][k]*(mhd->sy[i+1][j][k]+mhd->sy[i-1][j][k])
								                                                  +grid->ssy[i][j][k]*(mhd->sy[i][j+1][k]+mhd->sy[i][j-1][k])
								                                                  +grid->ssz[i][j][k]*(mhd->sy[i][j][k+1]+mhd->sy[i][j][k-1]);
							}
						}
					}
				}
			}
			#pragma omp section
			{
				for(i=1;i<grid->nxm1;i++)
				{
					for(j=1;j<grid->nym1;j++)
					{
						for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
						{
							if (mhd->szsmo[i][j][k] == 1)
							{
								mhd->sz[i][j][k]=grid->s[i][j][k]*mhd->sz[i][j][k]+grid->ssx[i][j][k]*(mhd->sz[i+1][j][k]+mhd->sz[i-1][j][k])
								                                                  +grid->ssy[i][j][k]*(mhd->sz[i][j+1][k]+mhd->sz[i][j-1][k])
								                                                  +grid->ssz[i][j][k]*(mhd->sz[i][j][k+1]+mhd->sz[i][j][k-1]);
							}
						}
					}
				}
			}
			#pragma omp section
			{
				for(i=1;i<grid->nxm1;i++)
				{
					for(j=1;j<grid->nym1;j++)
					{
						for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
						{
							if (mhd->sixsmo[i][j][k] == 1)
							{
								mhd->six[i][j][k]=grid->s[i][j][k]*mhd->six[i][j][k]+grid->ssx[i][j][k]*(mhd->six[i+1][j][k]+mhd->six[i-1][j][k])
								                                                    +grid->ssy[i][j][k]*(mhd->six[i][j+1][k]+mhd->six[i][j-1][k])
								                                                    +grid->ssz[i][j][k]*(mhd->six[i][j][k+1]+mhd->six[i][j][k-1]);
							}
						}
					}
				}
			}
			#pragma omp section
			{
				for(i=1;i<grid->nxm1;i++)
				{
					for(j=1;j<grid->nym1;j++)
					{
						for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
						{
							if (mhd->siysmo[i][j][k] == 1)
							{
								mhd->siy[i][j][k]=grid->s[i][j][k]*mhd->siy[i][j][k]+grid->ssx[i][j][k]*(mhd->siy[i+1][j][k]+mhd->siy[i-1][j][k])
								                                                    +grid->ssy[i][j][k]*(mhd->siy[i][j+1][k]+mhd->siy[i][j-1][k])
								                                                    +grid->ssz[i][j][k]*(mhd->siy[i][j][k+1]+mhd->siy[i][j][k-1]);
							}
						}
					}
				}
			}
			#pragma omp section
			{
				for(i=1;i<grid->nxm1;i++)
				{
					for(j=1;j<grid->nym1;j++)
					{
						for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
						{
							if (mhd->sizsmo[i][j][k] == 1)
							{
								mhd->siz[i][j][k]=grid->s[i][j][k]*mhd->siz[i][j][k]+grid->ssx[i][j][k]*(mhd->siz[i+1][j][k]+mhd->siz[i-1][j][k])
								                                                    +grid->ssy[i][j][k]*(mhd->siz[i][j+1][k]+mhd->siz[i][j-1][k])
								                                                    +grid->ssz[i][j][k]*(mhd->siz[i][j][k+1]+mhd->siz[i][j][k-1]);
							}
						}
					}
				}
			}
			#pragma omp section
			{
				for(i=1;i<grid->nxm1;i++)
				{
					for(j=1;j<grid->nym1;j++)
					{
						for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
						{
							if (mhd->snxsmo[i][j][k] == 1)
							{
								mhd->snx[i][j][k]=grid->s[i][j][k]*mhd->snx[i][j][k]+grid->ssx[i][j][k]*(mhd->snx[i+1][j][k]+mhd->snx[i-1][j][k])
								                                                    +grid->ssy[i][j][k]*(mhd->snx[i][j+1][k]+mhd->snx[i][j-1][k])
								                                                    +grid->ssz[i][j][k]*(mhd->snx[i][j][k+1]+mhd->snx[i][j][k-1]);
							}
						}
					}
				}
			}
			#pragma omp section
			{
				for(i=1;i<grid->nxm1;i++)
				{
					for(j=1;j<grid->nym1;j++)
					{
						for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
						{
							if (mhd->snysmo[i][j][k] == 1)
							{
								mhd->sny[i][j][k]=grid->s[i][j][k]*mhd->sny[i][j][k]+grid->ssx[i][j][k]*(mhd->sny[i+1][j][k]+mhd->sny[i-1][j][k])
								                                                    +grid->ssy[i][j][k]*(mhd->sny[i][j+1][k]+mhd->sny[i][j-1][k])
								                                                    +grid->ssz[i][j][k]*(mhd->sny[i][j][k+1]+mhd->sny[i][j][k-1]);
							}
						}
					}
				}
			}
			#pragma omp section
			{
				for(i=1;i<grid->nxm1;i++)
				{
					for(j=1;j<grid->nym1;j++)
					{
						for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
						{
							if (mhd->snzsmo[i][j][k] == 1)
							{
								mhd->snz[i][j][k]=grid->s[i][j][k]*mhd->snz[i][j][k]+grid->ssx[i][j][k]*(mhd->snz[i+1][j][k]+mhd->snz[i-1][j][k])
								                                                    +grid->ssy[i][j][k]*(mhd->snz[i][j+1][k]+mhd->snz[i][j-1][k])
								                                                    +grid->ssz[i][j][k]*(mhd->snz[i][j][k+1]+mhd->snz[i][j][k-1]);
							}
						}
					}
				}
			}
		} /*-- End OMP Sections --*/
		//Divergence B Solver
//		#pragma omp barrier
//		if (tid == 0)
//		{
//			printf("............................%04.2f [s] -----\n",(omp_get_wtime()-stime)/TIMESCALE);
//			stime=omp_get_wtime();
//			printf("----- DIV(B) Solver");
//		}
//		#pragma omp barrier
		#pragma omp single nowait
		{
			if (grid->divbsolve)
			{
				divbsolver(mhd,grid);
			}
		} /*-- End of Single Section --*/
		//Perform Third (2nd timestep) step
//		#pragma omp barrier
//		if (tid == 0)
//		{
//			printf(".....................%04.2f [s] -----\n",(omp_get_wtime()-stime)/TIMESCALE);
//			stime=omp_get_wtime();
//			printf("----- Second Timestep");
//		}
//		#pragma omp barrier
		#pragma omp for
		for(i=1;i<grid->nxm1;i++)
		{
			for(j=1;j<grid->nym1;j++)
			{
				for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
				{
					//  Density Equations
					mhd->rho[i][j][k]=mhd->rho[i][j][k]-grid->dt*drho[i][j][k];
					mhd->rhoi[i][j][k]=mhd->rhoi[i][j][k]-grid->dt*drhoi[i][j][k]
									     +grid->dt*qci[i][j][k];
					mhd->rhon[i][j][k]=mhd->rhon[i][j][k]-grid->dt*drhon[i][j][k] 
									     +grid->dt*qcn[i][j][k];
					//  Momentum Equations
					mhd->sx[i][j][k]=mhd->sx[i][j][k]-grid->dt*dsx[i][j][k]
									 -grid->dt*( vxbx[i][j][k]
										    -vexbx[i][j][k]
										    -srho[i][j][k]*mhd->gravx[i][j][k]
										    -qsdx[i][j][k]
										    +memd*srho[i][j][k]*mhd->zd[i][j][k]*rhoeinv[i][j][k]*(srhoe[i][j][k]*mhd->gravx[i][j][k]+qsex[i][j][k]))
					                                 -grid->dt*cdsx[i][j][k];
					mhd->sy[i][j][k]=mhd->sy[i][j][k]-grid->dt*dsy[i][j][k]
									 -grid->dt*( vxby[i][j][k]
										    -vexby[i][j][k]
										    -srho[i][j][k]*mhd->gravy[i][j][k]
										    -qsdy[i][j][k]
										    +memd*srho[i][j][k]*mhd->zd[i][j][k]*rhoeinv[i][j][k]*(srhoe[i][j][k]*mhd->gravy[i][j][k]+qsey[i][j][k]))
					                                 -grid->dt*cdsy[i][j][k];
					mhd->sz[i][j][k]=mhd->sz[i][j][k]-grid->dt*dsz[i][j][k]
									 -grid->dt*( vxbz[i][j][k]
										    -vexbz[i][j][k]
										    -srho[i][j][k]*mhd->gravz[i][j][k]
										    -qsdz[i][j][k]
										    +memd*srho[i][j][k]*mhd->zd[i][j][k]*rhoeinv[i][j][k]*(srhoe[i][j][k]*mhd->gravz[i][j][k]+qsez[i][j][k]))
					                                   -grid->dt*cdsz[i][j][k];
					mhd->six[i][j][k]=mhd->six[i][j][k]-grid->dt*dsix[i][j][k]
									   +grid->dt*( ivixbx[i][j][k]
										      -ivexbx[i][j][k]
										      +srhoi[i][j][k]*mhd->gravx[i][j][k]
										      +qsix[i][j][k]
										      +memi*srhoi[i][j][k]*mhd->zi[i][j][k]*rhoeinv[i][j][k]*(srhoe[i][j][k]*mhd->gravx[i][j][k]+qsex[i][j][k]))
					                                   -grid->dt*cdsix[i][j][k];
					mhd->siy[i][j][k]=mhd->siy[i][j][k]-grid->dt*dsiy[i][j][k]
									   +grid->dt*( ivixby[i][j][k]
										      -ivexby[i][j][k]
										      +srhoi[i][j][k]*mhd->gravy[i][j][k]
										      +qsiy[i][j][k]
										      +memi*srhoi[i][j][k]*mhd->zi[i][j][k]*rhoeinv[i][j][k]*(srhoe[i][j][k]*mhd->gravy[i][j][k]+qsey[i][j][k]))
					                                   -grid->dt*cdsiy[i][j][k];
					mhd->siz[i][j][k]=mhd->siz[i][j][k]-grid->dt*dsiz[i][j][k]
									   +grid->dt*( ivixbz[i][j][k]
										      -ivexbz[i][j][k]
										      +srhoi[i][j][k]*mhd->gravz[i][j][k]
										      +qsiz[i][j][k]
										      +memi*srhoi[i][j][k]*mhd->zi[i][j][k]*rhoeinv[i][j][k]*(srhoe[i][j][k]*mhd->gravz[i][j][k]+qsez[i][j][k]))
					                                   -grid->dt*cdsiz[i][j][k];
					mhd->snx[i][j][k]=mhd->snx[i][j][k]-grid->dt*dsnx[i][j][k]-grid->dt*(-1.*srhon[i][j][k]*mhd->gravx[i][j][k]-qsnx[i][j][k]);
					mhd->sny[i][j][k]=mhd->sny[i][j][k]-grid->dt*dsny[i][j][k]-grid->dt*(-1.*srhon[i][j][k]*mhd->gravy[i][j][k]-qsny[i][j][k]);
					mhd->snz[i][j][k]=mhd->snz[i][j][k]-grid->dt*dsnz[i][j][k]-grid->dt*(-1.*srhon[i][j][k]*mhd->gravz[i][j][k]-qsnz[i][j][k]);
					//  Presure Equations
					mhd->p[i][j][k]=mhd->p[i][j][k]-grid->dt*dp[i][j][k]
								       +grid->dt*(mhd->gamma[i][j][k]-1.)*qed[i][j][k]
				                                       -grid->dt*cdpd[i][j][k];
					mhd->pi[i][j][k]=mhd->pi[i][j][k]-grid->dt*dpi[i][j][k]
									 +grid->dt*(mhd->gammai[i][j][k]-1.)*qei[i][j][k]
				                                         -grid->dt*cdpi[i][j][k];
					mhd->pe[i][j][k]=mhd->pe[i][j][k]-grid->dt*dpe[i][j][k]
									 +grid->dt*(mhd->gammae[i][j][k]-1.)*qee[i][j][k]
				                                         -grid->dt*cdpe[i][j][k];
					mhd->pn[i][j][k]=mhd->pn[i][j][k]-grid->dt*dpn[i][j][k]
									 +grid->dt*(mhd->gamman[i][j][k]-1.)*qen[i][j][k]
				                                         -grid->dt*cdpn[i][j][k];
					//  Induction Equations
					mhd->bx[i][j][k]=mhd->bx[i][j][k]+grid->dt*dbx[i][j][k];
					mhd->by[i][j][k]=mhd->by[i][j][k]+grid->dt*dby[i][j][k];
					mhd->bz[i][j][k]=mhd->bz[i][j][k]+grid->dt*dbz[i][j][k];
				}
			}
		}
		//Update Electron Continuity (removed from previous for loop for parallelization due to rho reference.
		#pragma omp for
		for(i=1;i<grid->nxm1;i++)
		{
			for(j=1;j<grid->nym1;j++)
			{
				for(k=1+(i+j+gchk)%2;k<grid->nzm1;k=k+2)
				{
					mhd->rhoe[i][j][k]= (1-grid->econt)*mhd->me*(mhd->zi[i][j][k]*mhd->rhoi[i][j][k]*miinv-mhd->zd[i][j][k]*mhd->rho[i][j][k]*mdinv)
					                   +grid->econt*(mhd->rhoe[i][j][k]-grid->dt*drhoe[i][j][k]);
				}
			}
		}
//		if (tid == 0)
//		{
//			printf("...................%04.2f [s] -----\n",(omp_get_wtime()-stime)/TIMESCALE);
//			printf("***** End of Leap *****\n");
//		}
	} /*-- End of Parallel region --*/
}
//*****************************************************************************
/*
	Function:	smooth
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		10/23/07
	Inputs:		field, tsmooth
	Outputs:	none
	Purpose:	Checks field for smoothing and returns smoothing
			value in tsmooth.  The double data type in
			C has approximately 15 significant figures.  

                        mtemp= (x(i+1)-x)*(x-x(i-1)) ~ dx^2

*/
void smooth(double ***field,unsigned short int ***tsmooth,GRID *grid,int tid)
{
	int i,j,k;
	/*double mtempx[grid->nx][grid->ny][grid->nz];
	double mtempy[grid->nx][grid->ny][grid->nz];
	double mtempz[grid->nx][grid->ny][grid->nz];*/
	double ***mtempx;
	double ***mtempy;
	double ***mtempz;
	double mtempinv;
        double etol=1e-10;  //Oscillation amplitude tollerance
        double itol=1e-2;   //Inverse tollerance
	
	// Allocate mtemps
	mtempx=(double***)newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mtempy=(double***)newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mtempz=(double***)newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	// Initilize Arrays
	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				mtempx[i][j][k]=0.0;
				mtempy[i][j][k]=0.0;
				mtempz[i][j][k]=0.0;
				tsmooth[i][j][k]=0;
			}
		}
	}
	// Calc arrays
	for(i=1;i<grid->nx-1;i++)
	{
		for(j=1;j<grid->ny-1;j++)
		{
			for(k=1;k<grid->nz-1;k++)
			{
				mtempinv=1.;
				if (sqrt(field[i][j][k]*field[i][j][k]) > itol) mtempinv=1./field[i][j][k];
				mtempx[i][j][k]=(field[i+1][j][k]-field[i][j][k])*(field[i][j][k]-field[i-1][j][k]);
				mtempy[i][j][k]=(field[i][j+1][k]-field[i][j][k])*(field[i][j][k]-field[i][j-1][k]);
				mtempz[i][j][k]=(field[i][j][k+1]-field[i][j][k])*(field[i][j][k]-field[i][j][k-1]);
				if (sqrt(mtempx[i][j][k]*mtempx[i][j][k])*mtempinv*mtempinv <= etol) mtempx[i][j][k]=0.0;
				if (sqrt(mtempy[i][j][k]*mtempy[i][j][k])*mtempinv*mtempinv <= etol) mtempy[i][j][k]=0.0;
				if (sqrt(mtempz[i][j][k]*mtempz[i][j][k])*mtempinv*mtempinv <= etol) mtempz[i][j][k]=0.0;
			}
		}
	}
	// Check for Gridscale oscillations.
	for(i=1;i<grid->nx-1;i++)
	{
		for(j=1;j<grid->ny-1;j++)
		{
			for(k=1;k<grid->nz-1;k++)
			{
				if (   ((mtempx[i][j][k] < 0.0) && ((mtempx[i+1][j][k] < 0.0)||(mtempx[i-1][j][k] < 0.0))) 
				    || ((mtempy[i][j][k] < 0.0) && ((mtempy[i][j+1][k] < 0.0)||(mtempy[i][j-1][k] < 0.0)))
				    || ((mtempz[i][j][k] < 0.0) && ((mtempz[i][j][k+1] < 0.0)||(mtempx[i][j][k-1] < 0.0)))) tsmooth[i][j][k]=1;
			}
		}
	}
	// Free Allocations
	free(mtempx);
	free(mtempy);
	free(mtempz);
}
//*****************************************************************************
/*
	Function:	valcheck
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		9/29/08
	Inputs:		MHD mhd,GRID grid,n
	Outputs:	none
	Purpose:	Checks for valid values.  If not valid then it checks
			grid->chksmooth to see if values should be smoothed.
			Else it ends the program at a bad value.

*/
void valcheck(MHD *mhd,GRID *grid,int *n)
{
	double minrho,minrhoi,ne,maxvel,minp,minpi,minpe,vel;
	double minrhon,minpn;
	double mdinv,miinv,meinv;
	int i,j,k,nbadpoints;

//	Helpers
	mdinv=1./mhd->md;
	miinv=1./mhd->mi;
	meinv=1./mhd->me;
//      Define extremal values
	minrho=1.0e-10;
	minrhoi=1.0e-10;
	minrhon=1.0e-10;
	maxvel=10000.;
	minp=0.0000001;
	minpi=0.0000001;
	minpe=0.0000001;
	minpn=0.0000001;
	vel=0.;
	nbadpoints=0;
//      Check for electron Density
	for (i=1;i<grid->nxm1;i++)
	{
		for (j=1;j<grid->nym1;j++)
		{
			for (k=1;k<grid->nzm1;k++)
			{
				ne=mhd->zi[i][j][k]*mhd->rhoi[i][j][k]*miinv-mhd->zd[i][j][k]*mhd->rho[i][j][k]*mdinv;
				if ((grid->chksmooth) && (ne <=0.0))
				{
					mhd->rho[i][j][k]=(mhd->rho[i][j][k]+sqrt(mhd->rho[i+1][j][k]*mhd->rho[i+1][j][k])+sqrt(mhd->rho[i-1][j][k]*mhd->rho[i-1][j][k])+sqrt(mhd->rho[i][j+1][k]*mhd->rho[i][j+1][k])+sqrt(mhd->rho[i][j-1][k]*mhd->rho[i][j-1][k])+sqrt(mhd->rho[i][j][k+1]*mhd->rho[i][j][k+1])+sqrt(mhd->rho[i][j][k-1]*mhd->rho[i][j][k-1]))/7.;
					mhd->rhoi[i][j][k]=(mhd->rhoi[i][j][k]+sqrt(mhd->rhoi[i+1][j][k]*mhd->rhoi[i+1][j][k])+sqrt(mhd->rhoi[i-1][j][k]*mhd->rhoi[i-1][j][k])+sqrt(mhd->rhoi[i][j+1][k]*mhd->rhoi[i][j+1][k])+sqrt(mhd->rhoi[i][j-1][k]*mhd->rhoi[i][j-1][k])+sqrt(mhd->rhoi[i][j][k+1]*mhd->rhoi[i][j][k+1])+sqrt(mhd->rhoi[i][j][k-1]*mhd->rhoi[i][j][k-1]))/7.;
					mhd->rhoe[i][j][k]=mhd->me*(mhd->zi[i][j][k]*mhd->rhoi[i][j][k]/mhd->mi-mhd->zd[i][j][k]*mhd->rho[i][j][k]);
					nbadpoints++;
				}
				else if (ne <=0.0)
				{
					nbadpoints++;
				}
			}
		}
	}
	if (nbadpoints != 0)
	{
		printf(" ---Negative Electron Density (n=%d)\n",*n);
		printf("       at %d out of %d points\n",nbadpoints,grid->nx*grid->ny*grid->nz);
		if (grid->chksmooth)
		{
			printf(" ---Densities Smoothed at %d points\n",nbadpoints);
		}
		else
		{
			*n=-1;
		}
	}
//     Check Dust Density
	nbadpoints=0;
	for (i=1;i<grid->nxm1;i++)
	{
		for (j=1;j<grid->nym1;j++)
		{
			for (k=1;k<grid->nzm1;k++)
			{
				if ((grid->chksmooth) && (mhd->rho[i][j][k] <= minrho))
				{
					mhd->rho[i][j][k]=(mhd->rho[i][j][k]+sqrt(mhd->rho[i+1][j][k]*mhd->rho[i+1][j][k])+sqrt(mhd->rho[i-1][j][k]*mhd->rho[i-1][j][k])+sqrt(mhd->rho[i][j+1][k]*mhd->rho[i][j+1][k])+sqrt(mhd->rho[i][j-1][k]*mhd->rho[i][j-1][k])+sqrt(mhd->rho[i][j][k+1]*mhd->rho[i][j][k+1])+sqrt(mhd->rho[i][j][k-1]*mhd->rho[i][j][k-1]))/7.;
					mhd->rhoi[i][j][k]=(mhd->rhoi[i][j][k]+sqrt(mhd->rhoi[i+1][j][k]*mhd->rhoi[i+1][j][k])+sqrt(mhd->rhoi[i-1][j][k]*mhd->rhoi[i-1][j][k])+sqrt(mhd->rhoi[i][j+1][k]*mhd->rhoi[i][j+1][k])+sqrt(mhd->rhoi[i][j-1][k]*mhd->rhoi[i][j-1][k])+sqrt(mhd->rhoi[i][j][k+1]*mhd->rhoi[i][j][k+1])+sqrt(mhd->rhoi[i][j][k-1]*mhd->rhoi[i][j][k-1]))/7.;
					mhd->rhoe[i][j][k]=mhd->me*(mhd->zi[i][j][k]*mhd->rhoi[i][j][k]/mhd->mi-mhd->zd[i][j][k]*mhd->rho[i][j][k]);
					nbadpoints++;
				}
				else if (mhd->rho[i][j][k] <= minrho)
				{
					nbadpoints++;
				}
			}
		}
	}
	if (nbadpoints != 0)
	{
		printf(" ---Dust Density too low (n=%d)\n",*n);
		printf("       at %d out of %d points\n",nbadpoints,grid->nx*grid->ny*grid->nz);
		if (grid->chksmooth)
		{
			printf(" ---Densities Smoothed at %d points\n",nbadpoints);
		}
		else
		{
			*n=-1;
		}
	}
//	Check Ion Density
	nbadpoints=0;
	for (i=1;i<grid->nxm1;i++)
	{
		for (j=1;j<grid->nym1;j++)
		{
			for (k=1;k<grid->nzm1;k++)
			{
				if ((grid->chksmooth) && (mhd->rhoi[i][j][k] <= minrhoi))
				{
					mhd->rho[i][j][k]=(mhd->rho[i][j][k]+sqrt(mhd->rho[i+1][j][k]*mhd->rho[i+1][j][k])+sqrt(mhd->rho[i-1][j][k]*mhd->rho[i-1][j][k])+sqrt(mhd->rho[i][j+1][k]*mhd->rho[i][j+1][k])+sqrt(mhd->rho[i][j-1][k]*mhd->rho[i][j-1][k])+sqrt(mhd->rho[i][j][k+1]*mhd->rho[i][j][k+1])+sqrt(mhd->rho[i][j][k-1]*mhd->rho[i][j][k-1]))/7.;
					mhd->rhoi[i][j][k]=(mhd->rhoi[i][j][k]+sqrt(mhd->rhoi[i+1][j][k]*mhd->rhoi[i+1][j][k])+sqrt(mhd->rhoi[i-1][j][k]*mhd->rhoi[i-1][j][k])+sqrt(mhd->rhoi[i][j+1][k]*mhd->rhoi[i][j+1][k])+sqrt(mhd->rhoi[i][j-1][k]*mhd->rhoi[i][j-1][k])+sqrt(mhd->rhoi[i][j][k+1]*mhd->rhoi[i][j][k+1])+sqrt(mhd->rhoi[i][j][k-1]*mhd->rhoi[i][j][k-1]))/7.;
					mhd->rhoe[i][j][k]=mhd->me*(mhd->zi[i][j][k]*mhd->rhoi[i][j][k]/mhd->mi-mhd->zd[i][j][k]*mhd->rho[i][j][k]);
					nbadpoints++;
				}
				else if (mhd->rhoi[i][j][k] <= minrho)
				{
					nbadpoints++;
				}
			}
		}
	}
	if (nbadpoints != 0)
	{
		printf(" ---Ion Density too low (n=%d)\n",*n);
		printf("       at %d out of %d points\n",nbadpoints,grid->nx*grid->ny*grid->nz);
		if (grid->chksmooth)
		{
			printf(" ---Densities Smoothed at %d points\n",nbadpoints);
		}
		else
		{
			*n=-1;
		}
	}
//	Check Neutral Density
	nbadpoints=0;
	for (i=1;i<grid->nxm1;i++)
	{
		for (j=1;j<grid->nym1;j++)
		{
			for (k=1;k<grid->nzm1;k++)
			{
				if ((grid->chksmooth) && (mhd->rhon[i][j][k] <= minrhon))
				{
					mhd->rhon[i][j][k]=(mhd->rhon[i][j][k]+sqrt(mhd->rhon[i+1][j][k]*mhd->rhon[i+1][j][k])+sqrt(mhd->rhon[i-1][j][k]*mhd->rhon[i-1][j][k])+sqrt(mhd->rhon[i][j+1][k]*mhd->rhon[i][j+1][k])+sqrt(mhd->rhon[i][j-1][k]*mhd->rhon[i][j-1][k])+sqrt(mhd->rhon[i][j][k+1]*mhd->rhon[i][j][k+1])+sqrt(mhd->rhon[i][j][k-1]*mhd->rhon[i][j][k-1]))/7.;
					nbadpoints++;
				}
				else if (mhd->rhon[i][j][k] <= minrhon)
				{
					nbadpoints++;
				}
			}
		}
	}
	if (nbadpoints != 0)
	{
		printf(" ---Neutral Density too low (n=%d)\n",*n);
		printf("       at %d out of %d points\n",nbadpoints,grid->nx*grid->ny*grid->nz);
		if (grid->chksmooth)
		{
			printf(" ---Densities Smoothed at %d points\n",nbadpoints);
		}
		else
		{
			*n=-1;
		}
	}
	#pragma omp parallel default(none) shared(mhd,maxvel,grid,nbadpoints,n) private(i,j,k,vel)
	{
//	Check Dust Velocity
	#pragma omp single
	nbadpoints=0;
	#pragma omp for reduction(+:nbadpoints)
	for (i=1;i<grid->nxm1;i++)
	{
		for (j=1;j<grid->nym1;j++)
		{
			for (k=1;k<grid->nzm1;k++)
			{
				vel=0.0;
				vel=sqrt(mhd->sx[i][j][k]*mhd->sx[i][j][k]+mhd->sy[i][j][k]*mhd->sy[i][j][k]+mhd->sz[i][j][k]*mhd->sz[i][j][k])/mhd->rho[i][j][k];
				if (vel >= maxvel) nbadpoints++;
			}
		}
	}
	#pragma omp master
	if (nbadpoints != 0)
	{
		printf(" ---Dust Velocity too High (n=%d)\n",*n);
		printf("       at %d out of %d points\n",nbadpoints,grid->nx*grid->ny*grid->nz);
		*n=-1;
	}
//	Check Ion Velocity
	#pragma omp single
	nbadpoints=0;
	#pragma omp for reduction(+:nbadpoints)
	for (i=1;i<grid->nxm1;i++)
	{
		for (j=1;j<grid->nym1;j++)
		{
			for (k=1;k<grid->nzm1;k++)
			{
				vel=0.0;
				vel=pow(mhd->six[i][j][k]*mhd->six[i][j][k]+mhd->siy[i][j][k]*mhd->siy[i][j][k]+mhd->siz[i][j][k]*mhd->siz[i][j][k],.5)/mhd->rhoi[i][j][k];
				if (vel >= maxvel) nbadpoints++;
			}
		}
	}
	#pragma omp master
	if (nbadpoints != 0)
	{
		printf(" ---Ion Velocity too High (n=%d)\n",*n);
		printf("       at %d out of %d points\n",nbadpoints,grid->nx*grid->ny*grid->nz);
		*n=-1;
	}
//	Check Neutral Velocity
	#pragma omp single
	nbadpoints=0;
	#pragma omp for reduction(+:nbadpoints)
	for (i=1;i<grid->nxm1;i++)
	{
		for (j=1;j<grid->nym1;j++)
		{
			for (k=1;k<grid->nzm1;k++)
			{
				vel=0.0;
				vel=pow(mhd->snx[i][j][k]*mhd->snx[i][j][k]+mhd->sny[i][j][k]*mhd->sny[i][j][k]+mhd->snz[i][j][k]*mhd->snz[i][j][k],.5)/mhd->rhon[i][j][k];
				if (vel >= maxvel) nbadpoints++;
			}
		}
	}
	#pragma omp master
	if (nbadpoints != 0)
	{
		printf(" ---Neutral Velocity too High (n=%d)\n",*n);
		printf("       at %d out of %d points\n",nbadpoints,grid->nx*grid->ny*grid->nz);
		*n=-1;
	}
	} /*-- End of Parallel Section --*/
//	Check Dust Pressure
	nbadpoints=0;
	for (i=1;i<grid->nxm1;i++)
	{
		for (j=1;j<grid->nym1;j++)
		{
			for (k=1;k<grid->nzm1;k++)
			{
				if ((grid->chksmooth) && (mhd->p[i][j][k] <= minp))
				{
					mhd->p[i][j][k]=(mhd->p[i][j][k]+sqrt(mhd->p[i+1][j][k]*mhd->p[i+1][j][k])+sqrt(mhd->p[i-1][j][k]*mhd->p[i-1][j][k])+sqrt(mhd->p[i][j+1][k]*mhd->p[i][j+1][k])+sqrt(mhd->p[i][j-1][k]*mhd->p[i][j-1][k])+sqrt(mhd->p[i][j][k+1]*mhd->p[i][j][k+1])+sqrt(mhd->p[i][j][k-1]*mhd->p[i][j][k-1]))/7.;
					nbadpoints++;
				}
				else if (mhd->p[i][j][k] <= minp)
				{
					nbadpoints++;
				}
			}
		}
	}
	if (nbadpoints != 0)
	{
		printf(" ---Dust Pressure too low (n=%d)\n",*n);
		printf("       at %d out of %d points\n",nbadpoints,grid->nx*grid->ny*grid->nz);
		if (grid->chksmooth)
		{
			printf(" ---Dust Pressure Smoothed at %d points\n",nbadpoints);
		}
		else
		{
			*n=-1;
		}
	}
//	Check Ion Pressure
	nbadpoints=0;
	for (i=1;i<grid->nxm1;i++)
	{
		for (j=1;j<grid->nym1;j++)
		{
			for (k=1;k<grid->nzm1;k++)
			{
				if ((grid->chksmooth) && (mhd->pi[i][j][k] <= minpi))
				{
					mhd->pi[i][j][k]=(mhd->pi[i][j][k]+sqrt(mhd->pi[i+1][j][k]*mhd->pi[i+1][j][k])+sqrt(mhd->pi[i-1][j][k]*mhd->pi[i-1][j][k])+sqrt(mhd->pi[i][j+1][k]*mhd->pi[i][j+1][k])+sqrt(mhd->pi[i][j-1][k]*mhd->pi[i][j-1][k])+sqrt(mhd->pi[i][j][k+1]*mhd->pi[i][j][k+1])+sqrt(mhd->pi[i][j][k-1]*mhd->pi[i][j][k-1]))/7.;
					nbadpoints++;
				}
				else if (mhd->pi[i][j][k] <= minpi)
				{
					nbadpoints++;
				}
			}
		}
	}
	if (nbadpoints != 0)
	{
		printf(" ---Ion Pressure too low (n=%d)\n",*n);
		printf("       at %d out of %d points\n",nbadpoints,grid->nx*grid->ny*grid->nz);
		if (grid->chksmooth)
		{
			printf(" ---Ion Pressure Smoothed at %d points\n",nbadpoints);
		}
		else
		{
			*n=-1;
		}
	}
//	Check Electron Pressure
	nbadpoints=0;
	for (i=1;i<grid->nxm1;i++)
	{
		for (j=1;j<grid->nym1;j++)
		{
			for (k=1;k<grid->nzm1;k++)
			{
				if ((grid->chksmooth) && (mhd->pe[i][j][k] <= minpe))
				{
					mhd->pe[i][j][k]=(mhd->pe[i][j][k]+sqrt(mhd->pe[i+1][j][k]*mhd->pe[i+1][j][k])+sqrt(mhd->pe[i-1][j][k]*mhd->pe[i-1][j][k])+sqrt(mhd->pe[i][j+1][k]*mhd->pe[i][j+1][k])+sqrt(mhd->pe[i][j-1][k]*mhd->pe[i][j-1][k])+sqrt(mhd->pe[i][j][k+1]*mhd->pe[i][j][k+1])+sqrt(mhd->pe[i][j][k-1]*mhd->pe[i][j][k-1]))/7.;
					nbadpoints++;
				}
				else if (mhd->pe[i][j][k] <= minpe)
				{
					nbadpoints++;
				}
			}
		}
	}
	if (nbadpoints != 0)
	{
		printf(" ---Electron Pressure too low (n=%d)\n",*n);
		printf("       at %d out of %d points\n",nbadpoints,grid->nx*grid->ny*grid->nz);
		if (grid->chksmooth)
		{
			printf(" ---Electron Pressure Smoothed at %d points\n",nbadpoints);
		}
		else
		{
			*n=-1;
		}
	}
//	Check Neutral Pressure
	nbadpoints=0;
	for (i=1;i<grid->nxm1;i++)
	{
		for (j=1;j<grid->nym1;j++)
		{
			for (k=1;k<grid->nzm1;k++)
			{
				if ((grid->chksmooth) && (mhd->pn[i][j][k] <= minpn))
				{
					mhd->pn[i][j][k]=(mhd->pn[i][j][k]+sqrt(mhd->pn[i+1][j][k]*mhd->pn[i+1][j][k])+sqrt(mhd->pn[i-1][j][k]*mhd->pn[i-1][j][k])+sqrt(mhd->pn[i][j+1][k]*mhd->pn[i][j+1][k])+sqrt(mhd->pn[i][j-1][k]*mhd->pn[i][j-1][k])+sqrt(mhd->pn[i][j][k+1]*mhd->pn[i][j][k+1])+sqrt(mhd->pn[i][j][k-1]*mhd->pn[i][j][k-1]))/7.;
					nbadpoints++;
				}
				else if (mhd->pn[i][j][k] <= minpn)
				{
					nbadpoints++;
				}
			}
		}
	}
	if (nbadpoints != 0)
	{
		printf(" ---Neutral Pressure too low (n=%d)\n",*n);
		printf("       at %d out of %d points\n",nbadpoints,grid->nx*grid->ny*grid->nz);
		if (grid->chksmooth)
		{
			printf(" ---Neutral Pressure Smoothed at %d points\n",nbadpoints);
		}
		else
		{
			*n=-1;
		}
	}
}
//*****************************************************************************
/*
	Function:	divbsolver
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		9/18/08
	Inputs:		MHD mhd,GRID grid
	Outputs:	none
	Purpose:	Divergence B solver.  Based on work by Brackbill &
			Barnes, J. Comp. Phys. 35, 426--430 (5/1980).  Uses
			a SOR method (or Gauss-Seidel for SOR=1) to solve
			Poisson's equation.

			A Note on BC for phi (the unphysical potential)
			Periodic boundary conditions are implemented.
			Fixed Boundary Condition (zero gradient)
				phi[0,j,k]=phi[1,j,k]
				phi[1,j,k]=0.0
			Antisymetric Bx, Symmetric By
				phi[0,j,k]=phi[2,j,k]
				phi[1,j,k]=phi[1,j,k]
			Symmetric Bx, Antisymetric By
				phi[0,j,k]=-phi[2,j,k]
				phi[1,j,k]=-phi[1,j,k]

			At this time it is unclear if this procedure is
			effective.
			
*/
void divbsolver(MHD *mhd,GRID *grid)
{
	//double divb[grid->nx][grid->ny][grid->nz],phi[grid->nx][grid->ny][grid->nz],ophi[grid->nx][grid->ny][grid->nz];
	double ***divb;
	double ***phi;
	double ***ophi;
	double c0,SOR,tol,tol2;
	int i,j,k,redo,iter,maxiter;

	
	//Allocate Arrays
	divb = (double ***)newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	phi  = (double ***)newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	ophi = (double ***)newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	
//	Successive Overrelaxation Constant SOR>=1 (SOR=1 Gauss-Seidel)
	SOR=1.3;
//	Tollerance
	tol=1.0e-6;
	tol2=1.0e-8;
//	Maximum Iterations to find phi
	maxiter=50000;
	iter=0;
	// Initialize values
	for (i=0;i<grid->nx;i++)
	{
		for (j=0;j<grid->ny;j++)
		{
			for (k=0;k<grid->nz;k++)
			{
				divb[i][j][k]=0.0;
				phi[i][j][k]=0.0;
				ophi[i][j][k]=0.0;
			}
		}
	}
	// Calc Div(B)
	for (i=1;i<grid->nxm1;i++)
	{
		for (j=1;j<grid->nym1;j++)
		{
			for (k=1;k<grid->nzm1;k++)
			{
				divb[i][j][k]=(mhd->bx[i+1][j][k]-mhd->bx[i-1][j][k])*grid->difx[i][j][k]+(mhd->by[i][j+1][k]-mhd->by[i][j-1][k])*grid->dify[i][j][k]+(mhd->bz[i][j][k+1]-mhd->bz[i][j][k-1])*grid->difz[i][j][k];
				ophi[i][j][k]=divb[i][j][k];
			}
		}
	}
//	Solve for scalar potential
	do
	{
		printf("-----  Div(B) Solver iter:%6d\n",iter);
		iter++;
		redo=0;
		for (i=1;i<grid->nxm1;i++)
		{
			for (j=1;j<grid->nym1;j++)
			{
				for (k=1;k<grid->nzm1;k++)
				{
					c0=.5/(grid->ddifx[i][j][k]+grid->ddify[i][j][k]+grid->ddifz[i][j][k]);
					phi[i][j][k]=c0*((ophi[i+1][j][k]+phi[i-1][j][k])*grid->ddifx[i][j][k]+(ophi[i][j+1][k]+phi[i][j-1][k])*grid->ddify[i][j][k]+(ophi[i][j][k+1]+phi[i][j][k-1])*grid->ddifz[i][j][k]-divb[i][j][k]);
					phi[i][j][k]=ophi[i][j][k]+SOR*(phi[i][j][k]-ophi[i][j][k]);
				}
			}
		}
//		Apply BC
		for (j=0;j<grid->ny;j++)
		{
			for (k=0;k<grid->nz;k++)
			{
				if (grid->perx == 1)
				{
					phi[0][j][k]=phi[grid->nx][j][k];
					phi[grid->nx][j][k]=phi[1][j][k];
				}
				if (grid->perx ==0 )
				{
					phi[0][j][k]=0.0;
					phi[grid->nx][j][k]=0.0;
				}
			}
		}
		for (i=0;i<grid->nx;i++)
		{
			for (k=0;k<grid->nz;k++)
			{
				if (grid->pery == 1)
				{
					phi[i][0][k]=phi[i][grid->ny][k];
					phi[i][grid->ny][k]=phi[i][1][k];
				}
				if (grid->pery == 0)
				{
					phi[i][0][k]=0.0;
					phi[i][grid->ny][k]=0.0;
				}
			}
		}
		for (i=0;i<grid->nx;i++)
		{
			for (j=0;j<grid->ny;j++)
			{
				if (grid->perz == 1)
				{
					phi[i][j][0]=phi[i][j][grid->nz];
					phi[i][j][grid->nz]=phi[i][j][1];
				}
				if (grid->perz == 0)
				{
					phi[i][j][0]=0.0;
					phi[i][j][grid->nz]=0.0;
				}
			}
		}
		for (i=0;i<grid->nx;i++)
		{
			for (j=0;j<grid->ny;j++)
			{
				for (k=0;k<grid->nz;k++)
				{
					if ((abs(phi[i][j][k]-ophi[j][j][k]) >= tol2) && (fabs(divb[i][j][k]) > tol)) redo=1; 
					ophi[i][j][k]=phi[i][j][k];
				}
			}
		}
	} while (redo);
	printf("  -----  Divergence B solver called for %6d / %6d iterations\n",iter,maxiter);
//	printf(" ====Correcting B-Field====\n");
//	Subtract the potential from the field
	for (i=1;i<grid->nxm1;i++)
	{
		for (j=1;j<grid->nym1;j++)
		{
			for (k=1;k<grid->nzm1;k++)
			{
				mhd->bx[i][j][k]=mhd->bx[i][j][k]-(phi[i+1][j][k]-phi[i-1][j][k])*grid->difx[i][j][k];
				mhd->by[i][j][k]=mhd->by[i][j][k]-(phi[i][j+1][k]-phi[i][j-1][k])*grid->dify[i][j][k];
				mhd->bz[i][j][k]=mhd->bz[i][j][k]-(phi[i][j][k+1]-phi[i][j][k-1])*grid->difz[i][j][k];
			}
		}
	}
//	Recalc jx and sex
	for (i=1;i<grid->nxm1;i++)
	{
		for (j=1;j<grid->nym1;j++)
		{
			for (k=1;k<grid->nzm1;k++)
			{
				mhd->jx[i][j][k]=((mhd->bz[i][j+1][k]-mhd->bz[i][j-1][k])*grid->dify[i][j][k]-(mhd->by[i][j][k+1]-mhd->by[i][j][k-1])*grid->difz[i][j][k]);
				mhd->jy[i][j][k]=((mhd->bx[i][j][k+1]-mhd->bx[i][j][k-1])*grid->difz[i][j][k]-(mhd->bz[i+1][j][k]-mhd->bz[i-1][j][k])*grid->difx[i][j][k]);
				mhd->jz[i][j][k]=((mhd->by[i+1][j][k]-mhd->by[i-1][j][k])*grid->difx[i][j][k]-(mhd->bx[i][j+1][k]-mhd->bx[i][j-1][k])*grid->dify[i][j][k]);
				mhd->sex[i][j][k]=(mhd->me/mhd->mi)*mhd->zi[i][j][k]*mhd->six[i][j][k]-(mhd->me/mhd->md)*mhd->zd[i][j][k]*mhd->sx[i][j][k]-mhd->me*mhd->jx[i][j][k];
				mhd->sey[i][j][k]=(mhd->me/mhd->mi)*mhd->zi[i][j][k]*mhd->siy[i][j][k]-(mhd->me/mhd->md)*mhd->zd[i][j][k]*mhd->sy[i][j][k]-mhd->me*mhd->jy[i][j][k];
				mhd->sez[i][j][k]=(mhd->me/mhd->mi)*mhd->zi[i][j][k]*mhd->siz[i][j][k]-(mhd->me/mhd->md)*mhd->zd[i][j][k]*mhd->sz[i][j][k]-mhd->me*mhd->jz[i][j][k];
			}
		}
	}
	// Free Arrays
	free(divb);
	free(phi);
	free(ophi);
}

//*****************************************************************************
/*
	Function:	nudi
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		9/18/08
	Inputs:		MHD mhd
	Outputs:	none
	Purpose:	Dynamically calcs the dust ion collision frequency.
			   Zd*e^2         Ti         nd*Zd
			z=--------   tau=----     P=-------
			    a*Te          Te          ne
			      wpi^2*a
			nuch=---------------*(1+tau+z)
			      vti*sqrt(s*pi)
*/
void nudicalc(MHD *mhd,GRID *grid)
{
	int i,j,k;
	double wpi,z,tau,P,nuch,vti,lambda;

	double a=1.0e-6;
	double miinv=1./mhd->mi;
	double meinv=1./mhd->me;

	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				z=mhd->zd[i][j][k]/a/(mhd->pe[i][j][k]/mhd->rhoe[i][j][k]*mhd->me);
				tau=mhd->pi[i][j][k]*mhd->rhoe[i][j][k]/mhd->pe[i][j][k]/mhd->rhoi[i][j][k]*mhd->mi*meinv;
				P=mhd->rho[i][j][k]*mhd->me*mhd->zd[i][j][k]/mhd->rhoe[i][j][k];
				wpi=sqrt(mhd->rhoi[i][j][k]*miinv);
				vti=sqrt(mhd->pi[i][j][k]/mhd->rhoi[i][j][k]);
				nuch=wpi*wpi*a*(1+tau+z)/vti;
				lambda=sqrt(mhd->pi[i][j][k]*mhd->mi*mhd->mi/mhd->rhoi[i][j][k]/mhd->rhoi[i][j][k]);
				mhd->nuid[i][j][k]=nuch*2.*P*log(lambda/a)/(3*z*(1+tau+z)*tau*(1+P));
			}
		}
	}
}

//*****************************************************************************
/*
	Function:	nude
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		9/18/08
	Inputs:		MHD mhd
	Outputs:	none
	Purpose:	Dynamically calcs the dust ion collision frequency.
			   Zd*e^2         Ti         nd*Zd
			z=--------   tau=----     P=-------
			    a*Te          Te          ne
			      wpi^2*a
			nuch=---------------*(1+tau+z)
			      vti*sqrt(s*pi)
*/
void nudecalc(MHD *mhd,GRID *grid)
{
	int i,j,k;
	double wpi,z,tau,P,nuch,vti,lambda;

	double a=1.0e-6;
	double miinv=1./mhd->mi;
	double meinv=1./mhd->me;

	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				z=mhd->zd[i][j][k]/a/(mhd->pe[i][j][k]/mhd->rhoe[i][j][k]*mhd->me);
				tau=mhd->pi[i][j][k]*mhd->rhoe[i][j][k]/mhd->pe[i][j][k]/mhd->rhoi[i][j][k]*mhd->mi*meinv;
				P=mhd->rho[i][j][k]*mhd->me*mhd->zd[i][j][k]/mhd->rhoe[i][j][k];
				wpi=sqrt(mhd->rhoi[i][j][k]*miinv);
				vti=sqrt(mhd->pi[i][j][k]/mhd->rhoi[i][j][k]);
				nuch=wpi*wpi*a*(1+tau+z)/vti;
				lambda=sqrt(mhd->pi[i][j][k]*mhd->mi*mhd->mi/mhd->rhoi[i][j][k]/mhd->rhoi[i][j][k]);
				mhd->nude[i][j][k]=nuch*2.*P*exp(z)*(tau+z)*log(lambda/a)/(3*z*(1+tau+z))*mhd->rhoe[i][j][k]/mhd->rho[i][j][k];
			}
		}
	}
}

//*****************************************************************************
/*
	Function:	nuie
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		9/18/08
	Inputs:		MHD mhd
	Outputs:	none
	Purpose:	Dynamically calcs the ion electron collision freq.
			This is done through the assumption that the
			ion-electron collisional cross section is 10 times
			the neutron collisional cross section (5e-11).
*/
void nuiecalc(MHD *mhd,GRID *grid)
{
	int i,j,k;
	double vte;

	double meinv=1./mhd->me;

	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				vte=sqrt(mhd->pe[i][j][k]/mhd->rhoe[i][j][k]);
				mhd->nuie[i][j][k]=(mhd->rhoe[i][j][k]*meinv)/(mhd->me*mhd->me*vte*vte*vte);
			}
		}
	}
}

//*****************************************************************************
/*
	Function:	nudn
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		9/26/08
	Inputs:		MHD mhd
	Outputs:	none
	Purpose:	Dynamically calcs the dust neutral collision freq.
*/
void nudncalc(MHD *mhd,GRID *grid)
{
	int i,j,k;
	double vte;
	double mninv=1./mhd->mn;
	double sigma=1e-11;

	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				vte=sqrt(mhd->p[i][j][k]/mhd->rho[i][j][k]);
				mhd->nudn[i][j][k]=mninv*mhd->rhon[i][j][k]*sigma*vte;
			}
		}
	}
}

//*****************************************************************************
/*
	Function:	nuin
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		9/26/08
	Inputs:		MHD mhd
	Outputs:	none
	Purpose:	Dynamically calcs the ion neutral collision freq.
*/
void nuincalc(MHD *mhd,GRID *grid)
{
	int i,j,k;
	double vte;
	double mninv=1./mhd->mn;
	double sigma=1e-11;

	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				vte=sqrt(mhd->mi*mhd->pi[i][j][k]/mhd->rhoi[i][j][k]);
				mhd->nudn[i][j][k]=mninv*mhd->rhon[i][j][k]*sigma*vte;
			}
		}
	}
}

//*****************************************************************************
/*
	Function:	nuen
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		9/26/08
	Inputs:		MHD mhd
	Outputs:	none
	Purpose:	Dynamically calcs the electron neutral collision freq.
*/
void nuencalc(MHD *mhd,GRID *grid)
{
	int i,j,k;
	double vte;
	double mninv=1./mhd->mn;
	double sigma=1e-11;

	for(i=0;i<grid->nx;i++)
	{
		for(j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				vte=sqrt(mhd->me*mhd->pe[i][j][k]/mhd->rhoe[i][j][k]);
				mhd->nudn[i][j][k]=mninv*mhd->rhon[i][j][k]*sigma*vte;
			}
		}
	}
}

//*****************************************************************************
/*
	Function:	output
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		9/26/08
	Inputs:		MHD mhd,GRID grid,n,t
	Outputs:	none
	Purpose:	This fuction calculates the box edge values by averaging
			and outputs the dataset in the netCDF data format:
			http://www.unidata.ucar.edu/software/netcdf/
			The type value is used to help other programs
			(namely read_mhdust_netcdf) determine the format of the
			data.  At the time of this codeing we expect:
			1 = MHDust output
			2 = nMHDust output
*/
void output(MHD *mhd, GRID *grid,int n, double t, char *runname)
{
	FILE *fp;
	char	*filename;
	char	*tempc;
	int	tempd,i,j,k,test;
	int status,ncid,nx_dim,ny_dim,nz_dim,grid_dimids[3];
	int bc1_dim,bc2_dim,bc2k_dim,bc_dimids[2],bck_dimids[2];
	int rhoid,rhoiid,rhoeid,rhonid,pid,piid,peid,pnid;
	int sxid,syid,szid,sixid,siyid,sizid,sexid,seyid,sezid,snxid,snyid,snzid;
	int bxid,byid,bzid,jxid,jyid,jzid,exid,eyid,ezid;
	int zdid,ziid,gammaid,gammaiid,gammaeid,gammanid;
	int nuidid,nuieid,nudeid,nudnid,nuinid,nuenid;
	int gravxid,gravyid,gravzid;
	int visxid,visyid,viszid,mdid,miid,meid,mnid,ionizid,recomid,eid;
	int xid,yid,zid,timeid,xminid,xmaxid,yminid,ymaxid,zminid,zmaxid,typeid;
	int difxid,difyid,difzid,ddifxid,ddifyid,ddifzid,ddifmxid,ddifmyid,ddifmzid;
	int ddifpxid,ddifpyid,ddifpzid,meanmxid,meanmyid,meanmzid,meanpxid,meanpyid,meanpzid;
	int econtid,perxid,peryid,perzid,divbsolveid,movieoutid,ballonid,lastballid;
	int dnudiid,dnudeid,dnuieid,dnudnid,dnuinid,dnuenid,dtid;
	int kid,aid,did,c1id,c2id,c3aid,c3bid,noutid,maxntimeid;
	static size_t start[] = {0,0,0}; /* start at first value */
	static size_t start2d[] = {0, 0};
	size_t count[] = {grid->nx,grid->ny,grid->nz};
	static size_t countk2d[] = {6, 9};
	static size_t count2d[] = {6, 20};
	static type=2;	/*1=MHDust; 2=nMHDust*/


//	Create the filename
	filename=(char*)malloc(sizeof(char)*strlen(runname));
	tempc=(char*)malloc(6*sizeof(char));    // Allocate tempc
	sprintf(tempc,"%06d",n);		// Get the iteration as a string
	strcpy(filename,runname);		// Set filename=runname
	strcat(filename,tempc);			// Add the iteration to the filename
	strcat(filename,".nc");			// Add the .nc extension
	if (n == -1) filename="dataerror.nc";	// Handle an error file

//      Begin netCDF section
	/* create netCDF dataset: enter define mode */
	status = nc_create(filename,NC_64BIT_OFFSET,&ncid);
	if (status != NC_NOERR) handle_error(status);
	/* define dimensions: from name and length */
	status = nc_def_dim(ncid,"NX",grid->nx,&nx_dim);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_dim(ncid,"NY",grid->ny,&ny_dim);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_dim(ncid,"NZ",grid->nz,&nz_dim);
	if (status != NC_NOERR) handle_error(status);
        status = nc_def_dim(ncid,"BC1",6,&bc1_dim);
	if (status != NC_NOERR) handle_error(status);
        status = nc_def_dim(ncid,"BC2",20,&bc2_dim);
	if (status != NC_NOERR) handle_error(status);
        status = nc_def_dim(ncid,"BC2K",9,&bc2k_dim);
	if (status != NC_NOERR) handle_error(status);
	/* define variables: from name, type, ... */
	grid_dimids[0] = nx_dim;
	grid_dimids[1] = ny_dim;
	grid_dimids[2] = nz_dim;
	bc_dimids[0] = bc1_dim;
	bc_dimids[1] = bc2_dim;
	bck_dimids[0] = bc1_dim;
	bck_dimids[1] = bc2k_dim;
	status = nc_def_var(ncid,"rho",NC_DOUBLE,3,grid_dimids,&rhoid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"rhoi",NC_DOUBLE,3,grid_dimids,&rhoiid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"rhoe",NC_DOUBLE,3,grid_dimids,&rhoeid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"rhon",NC_DOUBLE,3,grid_dimids,&rhonid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"p",NC_DOUBLE,3,grid_dimids,&pid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"pi",NC_DOUBLE,3,grid_dimids,&piid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"pe",NC_DOUBLE,3,grid_dimids,&peid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"pn",NC_DOUBLE,3,grid_dimids,&pnid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"sx",NC_DOUBLE,3,grid_dimids,&sxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"sy",NC_DOUBLE,3,grid_dimids,&syid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"sz",NC_DOUBLE,3,grid_dimids,&szid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"six",NC_DOUBLE,3,grid_dimids,&sixid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"siy",NC_DOUBLE,3,grid_dimids,&siyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"siz",NC_DOUBLE,3,grid_dimids,&sizid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"sex",NC_DOUBLE,3,grid_dimids,&sexid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"sey",NC_DOUBLE,3,grid_dimids,&seyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"sez",NC_DOUBLE,3,grid_dimids,&sezid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"snx",NC_DOUBLE,3,grid_dimids,&snxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"sny",NC_DOUBLE,3,grid_dimids,&snyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"snz",NC_DOUBLE,3,grid_dimids,&snzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"bx",NC_DOUBLE,3,grid_dimids,&bxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"by",NC_DOUBLE,3,grid_dimids,&byid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"bz",NC_DOUBLE,3,grid_dimids,&bzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"jx",NC_DOUBLE,3,grid_dimids,&jxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"jy",NC_DOUBLE,3,grid_dimids,&jyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"jz",NC_DOUBLE,3,grid_dimids,&jzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"ex",NC_DOUBLE,3,grid_dimids,&exid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"ey",NC_DOUBLE,3,grid_dimids,&eyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"ez",NC_DOUBLE,3,grid_dimids,&ezid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"zd",NC_DOUBLE,3,grid_dimids,&zdid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"zi",NC_DOUBLE,3,grid_dimids,&ziid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"gamma",NC_DOUBLE,3,grid_dimids,&gammaid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"gammai",NC_DOUBLE,3,grid_dimids,&gammaiid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"gammae",NC_DOUBLE,3,grid_dimids,&gammaeid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"gamman",NC_DOUBLE,3,grid_dimids,&gammanid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"nuid",NC_DOUBLE,3,grid_dimids,&nuidid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"nuie",NC_DOUBLE,3,grid_dimids,&nuieid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"nude",NC_DOUBLE,3,grid_dimids,&nudeid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"nudn",NC_DOUBLE,3,grid_dimids,&nudnid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"nuin",NC_DOUBLE,3,grid_dimids,&nuinid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"nuen",NC_DOUBLE,3,grid_dimids,&nuenid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"gravx",NC_DOUBLE,3,grid_dimids,&gravxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"gravy",NC_DOUBLE,3,grid_dimids,&gravyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"gravz",NC_DOUBLE,3,grid_dimids,&gravzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"visx",NC_DOUBLE,0,grid_dimids,&visxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"visy",NC_DOUBLE,0,grid_dimids,&visyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"visz",NC_DOUBLE,0,grid_dimids,&viszid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"e",NC_DOUBLE,0,grid_dimids,&eid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"md",NC_DOUBLE,0,grid_dimids,&mdid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"mi",NC_DOUBLE,0,grid_dimids,&miid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"me",NC_DOUBLE,0,grid_dimids,&meid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"mn",NC_DOUBLE,0,grid_dimids,&mnid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"ioniz",NC_DOUBLE,0,grid_dimids,&ionizid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"recom",NC_DOUBLE,0,grid_dimids,&recomid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"time",NC_DOUBLE,0,grid_dimids,&timeid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"dt",NC_DOUBLE,0,grid_dimids,&dtid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"x",NC_DOUBLE,3,grid_dimids,&xid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"y",NC_DOUBLE,3,grid_dimids,&yid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"z",NC_DOUBLE,3,grid_dimids,&zid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"xmin",NC_DOUBLE,0,grid_dimids,&xminid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"xmax",NC_DOUBLE,0,grid_dimids,&xmaxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"ymin",NC_DOUBLE,0,grid_dimids,&yminid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"ymax",NC_DOUBLE,0,grid_dimids,&ymaxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"zmin",NC_DOUBLE,0,grid_dimids,&zminid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"zmax",NC_DOUBLE,0,grid_dimids,&zmaxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"type",NC_INT,0,grid_dimids,&typeid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"difx",NC_DOUBLE,3,grid_dimids,&difxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"dify",NC_DOUBLE,3,grid_dimids,&difyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"difz",NC_DOUBLE,3,grid_dimids,&difzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"ddifx",NC_DOUBLE,3,grid_dimids,&ddifxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"ddify",NC_DOUBLE,3,grid_dimids,&ddifyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"ddifz",NC_DOUBLE,3,grid_dimids,&ddifzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"ddifmx",NC_DOUBLE,3,grid_dimids,&ddifmxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"ddifmy",NC_DOUBLE,3,grid_dimids,&ddifmyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"ddifmz",NC_DOUBLE,3,grid_dimids,&ddifmzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"ddifpx",NC_DOUBLE,3,grid_dimids,&ddifpxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"ddifpy",NC_DOUBLE,3,grid_dimids,&ddifpyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"ddifpz",NC_DOUBLE,3,grid_dimids,&ddifpzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"meanmx",NC_DOUBLE,3,grid_dimids,&meanmxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"meanmy",NC_DOUBLE,3,grid_dimids,&meanmyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"meanmz",NC_DOUBLE,3,grid_dimids,&meanmzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"meanpx",NC_DOUBLE,3,grid_dimids,&meanpxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"meanpy",NC_DOUBLE,3,grid_dimids,&meanpyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"meanpz",NC_DOUBLE,3,grid_dimids,&meanpzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"econt",NC_SHORT,0,grid_dimids,&econtid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"perx",NC_SHORT,0,grid_dimids,&perxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"pery",NC_SHORT,0,grid_dimids,&peryid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"perz",NC_SHORT,0,grid_dimids,&perzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"divbsolve",NC_SHORT,0,grid_dimids,&divbsolveid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"movieout",NC_INT,0,grid_dimids,&movieoutid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"ballon",NC_SHORT,0,grid_dimids,&ballonid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"lastball",NC_INT,0,grid_dimids,&lastballid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"dnudi",NC_SHORT,0,grid_dimids,&dnudiid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"dnude",NC_SHORT,0,grid_dimids,&dnudeid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"dnuie",NC_SHORT,0,grid_dimids,&dnuieid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"dnudn",NC_SHORT,0,grid_dimids,&dnudnid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"dnuin",NC_SHORT,0,grid_dimids,&dnuinid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"dnuen",NC_SHORT,0,grid_dimids,&dnuenid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"nout",NC_INT,0,grid_dimids,&noutid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"maxntime",NC_INT,0,grid_dimids,&maxntimeid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"k",NC_INT,2,bck_dimids,&kid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"a",NC_INT,2,bc_dimids,&aid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"d",NC_INT,2,bc_dimids,&did);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"c1",NC_INT,2,bc_dimids,&c1id);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"c2",NC_INT,2,bc_dimids,&c2id);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"c3a",NC_INT,2,bc_dimids,&c3aid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid,"c3b",NC_INT,2,bc_dimids,&c3bid);
	if (status != NC_NOERR) handle_error(status);
	/* put attribute: assign attribute values */
	status = nc_put_att_text(ncid,rhoid,"units",strlen("free"),"free");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,rhoiid,"units",strlen("free"),"free");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,rhoeid,"units",strlen("free"),"free");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,rhonid,"units",strlen("free"),"free");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,pid,"units",strlen("B*B/2mu0"),"B*B/2mu0");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,piid,"units",strlen("B*B/2mu0"),"B*B/2mu0");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,peid,"units",strlen("B*B/2mu0"),"B*B/2mu0");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,pnid,"units",strlen("B*B/2mu0"),"B*B/2mu0");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,sxid,"units",strlen("rho*vA"),"rho*vA");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,syid,"units",strlen("rho*vA"),"rho*vA");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,szid,"units",strlen("rho*vA"),"rho*vA");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,sixid,"units",strlen("rho*vA"),"rho*vA");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,siyid,"units",strlen("rho*vA"),"rho*vA");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,sizid,"units",strlen("rho*vA"),"rho*vA");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,sexid,"units",strlen("rho*vA"),"rho*vA");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,seyid,"units",strlen("rho*vA"),"rho*vA");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,sezid,"units",strlen("rho*vA"),"rho*vA");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,snxid,"units",strlen("rho*vA"),"rho*vA");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,snyid,"units",strlen("rho*vA"),"rho*vA");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,snzid,"units",strlen("rho*vA"),"rho*vA");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,bxid,"units",strlen("free"),"free");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,byid,"units",strlen("free"),"free");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,bzid,"units",strlen("free"),"free");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,jxid,"units",strlen("B/(mu0*L)"),"B/(mu0*L)");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,jyid,"units",strlen("B/(mu0*L)"),"B/(mu0*L)");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,jzid,"units",strlen("B/(mu0*L)"),"B/(mu0*L)");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,exid,"units",strlen("B*vA"),"B*vA");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,eyid,"units",strlen("B*vA"),"B*vA");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,ezid,"units",strlen("B*vA"),"B*vA");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,zdid,"units",strlen("Electron_Charges"),"Electron_Charges");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,ziid,"units",strlen("Positron_Charges"),"Positron_Charges");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,gammaid,"units",strlen(" ")," ");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,gammaiid,"units",strlen(" ")," ");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,gammaeid,"units",strlen(" ")," ");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,gammanid,"units",strlen(" ")," ");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,nuidid,"units",strlen("1/t"),"1/t");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,nuieid,"units",strlen("1/t"),"1/t");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,nudeid,"units",strlen("1/t"),"1/t");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,nudnid,"units",strlen("1/t"),"1/t");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,nuinid,"units",strlen("1/t"),"1/t");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,nuenid,"units",strlen("1/t"),"1/t");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,gravxid,"units",strlen("vA/t"),"vA/t");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,gravyid,"units",strlen("vA/t"),"vA/t");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,gravzid,"units",strlen("vA/t"),"vA/t");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,visxid,"units",strlen(" ")," ");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,visyid,"units",strlen(" ")," ");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,viszid,"units",strlen(" ")," ");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,mdid,"units",strlen("free"),"free");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,miid,"units",strlen("Dutst Masses"),"Dust Masses");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,meid,"units",strlen("Dust Masses"),"Dust Masses");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,mnid,"units",strlen("Dust Masses"),"Dust Masses");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,ionizid,"units",strlen("free"),"free");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,recomid,"u",strlen("free"),"free");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,timeid,"units",strlen("free"),"free");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,xid,"units",strlen("free"),"free");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,yid,"units",strlen("free"),"free");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,zid,"units",strlen("free"),"free");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,typeid,"Type",strlen("1=MHDust;2=nMHDust"),"1=MHDust;2=nMHDust");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,noutid,"Description",strlen("Output Interval"),"Output Interval");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,maxntimeid,"Description",strlen("Last Timestep"),"Last Timestep");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,NC_GLOBAL,"code",7,"nMHDust");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,NC_GLOBAL,"version",3,"1.0");
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_att_text(ncid,NC_GLOBAL,"creator",7,"nMHDust");
	if (status != NC_NOERR) handle_error(status);
	/* end definitions: leave define mode */
	status = nc_enddef(ncid);
	if (status != NC_NOERR) handle_error(status);
	/* provide values for variables */
	status = nc_put_vara(ncid,rhoid,start,count,&mhd->rho[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,rhoiid,start,count,&mhd->rhoi[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,rhoeid,start,count,&mhd->rhoe[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,rhonid,start,count,&mhd->rhon[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,pid,start,count,&mhd->p[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,piid,start,count,&mhd->pi[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,peid,start,count,&mhd->pe[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,pnid,start,count,&mhd->pn[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,sxid,start,count,&mhd->sx[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,syid,start,count,&mhd->sy[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,szid,start,count,&mhd->sz[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,sixid,start,count,&mhd->six[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,siyid,start,count,&mhd->siy[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,sizid,start,count,&mhd->siz[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,sexid,start,count,&mhd->sex[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,seyid,start,count,&mhd->sey[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,sezid,start,count,&mhd->sez[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,snxid,start,count,&mhd->snx[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,snyid,start,count,&mhd->sny[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,snzid,start,count,&mhd->snz[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,bxid,start,count,&mhd->bx[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,byid,start,count,&mhd->by[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,bzid,start,count,&mhd->bz[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,jxid,start,count,&mhd->jx[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,jyid,start,count,&mhd->jy[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,jzid,start,count,&mhd->jz[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,exid,start,count,&mhd->ex[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,eyid,start,count,&mhd->ey[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,ezid,start,count,&mhd->ez[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,zdid,start,count,&mhd->zd[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,ziid,start,count,&mhd->zi[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,gammaid,start,count,&mhd->gamma[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,gammaiid,start,count,&mhd->gammai[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,gammaeid,start,count,&mhd->gammae[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,gammanid,start,count,&mhd->gamman[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,nuidid,start,count,&mhd->nuid[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,nuieid,start,count,&mhd->nuie[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,nudeid,start,count,&mhd->nude[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,nudnid,start,count,&mhd->nudn[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,nuinid,start,count,&mhd->nuin[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,nuenid,start,count,&mhd->nuen[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,gravxid,start,count,&mhd->gravx[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,gravyid,start,count,&mhd->gravy[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,gravzid,start,count,&mhd->gravz[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_double(ncid,visxid,&(mhd->visx));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_double(ncid,visyid,&(mhd->visy));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_double(ncid,viszid,&(mhd->visz));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_double(ncid,eid,&(mhd->e));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_double(ncid,mdid,&(mhd->md));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_double(ncid,miid,&(mhd->mi));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_double(ncid,meid,&(mhd->me));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_double(ncid,mnid,&(mhd->mn));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_double(ncid,ionizid,&(mhd->ioniz));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_double(ncid,recomid,&(mhd->recom));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_double(ncid,timeid,&(t));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_double(ncid,dtid,&(grid->dt));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,xid,start,count,&grid->x[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,yid,start,count,&grid->y[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,zid,start,count,&grid->z[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_double(ncid,xminid,&(grid->xmin));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_double(ncid,xmaxid,&(grid->xmax));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_double(ncid,yminid,&(grid->ymin));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_double(ncid,ymaxid,&(grid->ymax));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_double(ncid,zminid,&(grid->zmin));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_double(ncid,zmaxid,&(grid->zmax));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_int(ncid,typeid,&(type));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,difxid,start,count,&grid->difx[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,difyid,start,count,&grid->dify[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,difzid,start,count,&grid->difz[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,ddifxid,start,count,&grid->ddifx[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,ddifyid,start,count,&grid->ddify[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,ddifzid,start,count,&grid->ddifz[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,ddifmxid,start,count,&grid->ddifmx[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,ddifmyid,start,count,&grid->ddifmy[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,ddifmzid,start,count,&grid->ddifmz[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,ddifpxid,start,count,&grid->ddifpx[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,ddifpyid,start,count,&grid->ddifpy[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,ddifpzid,start,count,&grid->ddifpz[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,meanmxid,start,count,&grid->meanmx[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,meanmyid,start,count,&grid->meanmy[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,meanmzid,start,count,&grid->meanmz[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,meanpxid,start,count,&grid->meanpx[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,meanpyid,start,count,&grid->meanpy[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,meanpzid,start,count,&grid->meanpz[0][0][0]);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_short(ncid,econtid,&(grid->econt));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_short(ncid,perxid,&(grid->perx));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_short(ncid,peryid,&(grid->pery));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_short(ncid,perzid,&(grid->perz));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_short(ncid,divbsolveid,&(grid->divbsolve));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_int(ncid,movieoutid,&(grid->movieout));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_short(ncid,ballonid,&(grid->ballon));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_int(ncid,lastballid,&(grid->lastball));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_short(ncid,dnudiid,&(grid->dnudi));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_short(ncid,dnudeid,&(grid->dnude));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_short(ncid,dnuieid,&(grid->dnuie));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_short(ncid,dnudnid,&(grid->dnudn));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_short(ncid,dnuinid,&(grid->dnuin));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_short(ncid,dnuenid,&(grid->dnuen));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_int(ncid,noutid,&(grid->nout));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_var_int(ncid,maxntimeid,&(grid->maxntime));
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,kid,start2d,countk2d,grid->k);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,aid,start2d,count2d,grid->a);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,did,start2d,count2d,grid->d);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,c1id,start2d,count2d,grid->c1);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,c2id,start2d,count2d,grid->c2);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,c3aid,start2d,count2d,grid->c3a);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara(ncid,c3bid,start2d,count2d,grid->c3b);
	if (status != NC_NOERR) handle_error(status);
	/* close: save new netCDF dataset */
	status = nc_close(ncid);
	if (status != NC_NOERR) handle_error(status);
//      End netCDF section
	if (grid->verb) printf(" --Output at %2.4f (n=%4d)\n",t,n);
//	Free Memory Allocation
	free(filename);
	free(tempc);
}
//*****************************************************************************
/*
	Function:	netcdf_input
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		10/27/10
	Inputs:		MHD mhd,GRID grid,filename
	Outputs:	none
	Purpose:	This fuction reads the dataset in the netCDF data format:
			http://www.unidata.ucar.edu/software/netcdf/
*/
void netcdf_input(MHD *mhd, GRID *grid,char *filename)
{
	FILE *fp;
	char	*tempc;
	int	tempd,i,j,k,test;
	int status,ncid,nx_dim,ny_dim,nz_dim,grid_dimids[3];
	int bc1_dim,bc2_dim,bc2k_dim,bc_dimids[2],bck_dimids[2];
	int rhoid,rhoiid,rhoeid,rhonid,pid,piid,peid,pnid;
	int sxid,syid,szid,sixid,siyid,sizid,sexid,seyid,sezid,snxid,snyid,snzid;
	int bxid,byid,bzid,jxid,jyid,jzid,exid,eyid,ezid;
	int zdid,ziid,gammaid,gammaiid,gammaeid,gammanid;
	int nuidid,nuieid,nudeid,nudnid,nuinid,nuenid;
	int gravxid,gravyid,gravzid;
	int visxid,visyid,viszid,mdid,miid,meid,mnid,ionizid,recomid,eid;
	int xid,yid,zid,timeid,xminid,xmaxid,yminid,ymaxid,zminid,zmaxid,typeid;
	int difxid,difyid,difzid,ddifxid,ddifyid,ddifzid,ddifmxid,ddifmyid,ddifmzid;
	int ddifpxid,ddifpyid,ddifpzid,meanmxid,meanmyid,meanmzid,meanpxid,meanpyid,meanpzid;
	int econtid,perxid,peryid,perzid,divbsolveid,divbintid,ballonid,lastballid;
	int dnudiid,dnudeid,dnuieid,dnudnid,dnuinid,dnuenid,dtid;
	int kid,aid,did,c1id,c2id,c3aid,c3bid,noutid,maxntimeid;
	static size_t start[] = {0,0,0}; /* start at first value */
	static size_t start2d[] = {0, 0};
	size_t count[] = {grid->nx,grid->ny,grid->nz};
	static size_t countk2d[] = {6, 9};
	static size_t count2d[] = {6, 20};
	static type=2;	/*1=MHDust; 2=nMHDust*/
	
	//Open the netCDF File
	status=nc_open(filename,NC_NOWRITE,&ncid);
	if (status != NC_NOERR) handle_error(status);
	//Get the dimenisons first and allocate them
	status = nc_inq_dimid(ncid, "NX", &nx_dim);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_dimid(ncid, "NY", &ny_dim);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_dimid(ncid, "NZ", &nz_dim);
	if (status != NC_NOERR) handle_error(status);
	//status = nc_inq_dimlen(ncid, nx_dim, &grid->nx);
	if (status != NC_NOERR) handle_error(status);
	//status = nc_inq_dimlen(ncid, ny_dim, &grid->ny);
	if (status != NC_NOERR) handle_error(status);
	//status = nc_inq_dimlen(ncid, nz_dim, &grid->nz);
	if (status != NC_NOERR) handle_error(status);
	allocate_grid(mhd,grid);
	//Inquire the ID's of the variables
	status = nc_inq_varid (ncid, "rho", &rhoid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "rhoi", &rhoiid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "rhoe", &rhoeid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "rhon", &rhonid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "p", &pid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "pi", &piid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "pe", &peid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "pn", &pnid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "sx", &sxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "sy", &syid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "sz", &szid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "six", &sixid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "siy", &siyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "siz", &sizid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "sex", &sexid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "sey", &seyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "sez", &sezid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "snx", &snxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "sny", &snyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "snz", &snzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "bx", &bxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "by", &byid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "bz", &bzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "ex", &exid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "ey", &eyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "ez", &ezid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "jx", &jxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "jy", &jyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "jz", &jzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "gamma", &gammaid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "gammai", &gammaiid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "gammae", &gammaeid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "gamman", &gammanid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "nuid", &nuidid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "nuie", &nuieid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "nude", &nudeid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "nudn", &nudnid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "nuin", &nuinid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "nuen", &nuenid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "gravx", &gravxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "gravy", &gravyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "gravz", &gravzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "visx", &visxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "visy", &visyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "visz", &viszid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "zd", &zdid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "zi", &ziid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "md", &mdid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "mi", &miid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "me", &meid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "mn", &mnid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "ioniz", &ionizid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "recom", &recomid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "e", &eid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "x", &xid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "y", &yid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "z", &zid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "time", &timeid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "dt", &dtid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "xmin", &xminid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "xmax", &xmaxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "ymin", &yminid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "ymax", &ymaxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "zmin", &zminid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "zmax", &zmaxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "difx", &difxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "dify", &difyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "difz", &difzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "ddifx", &ddifxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "ddify", &ddifyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "ddifz", &ddifzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "ddifmx", &ddifmxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "ddifmy", &ddifmyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "ddifmz", &ddifmzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "ddifpx", &ddifpxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "ddifpy", &ddifpyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "ddifpz", &ddifpzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "meanmx", &meanmxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "meanmy", &meanmyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "meanmz", &meanmzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "meanpx", &meanpxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "meanpy", &meanpyid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "meanpz", &meanpzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "econt", &econtid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "perx", &perxid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "pery", &peryid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "perz", &perzid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "divbsolve", &divbsolveid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "dibint", &divbintid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "ballon", &ballonid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "lastball", &lastballid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "dnudi", &dnudiid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "dnude", &dnudeid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "dnuie", &dnuieid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "dnudn", &dnudnid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "dnuin", &dnuinid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "dnuen", &dnuenid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "nout", &noutid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "maxntime", &maxntimeid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "k", &kid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "a", &aid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "d", &did);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "c1", &c1id);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "c2", &c2id);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "c3a", &c3aid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "c3b", &noutid);
	if (status != NC_NOERR) handle_error(status);
	//Close the netCDF File
	status=nc_close(ncid);
	if (status != NC_NOERR) handle_error(status);
}
//*****************************************************************************
/*
	Function:	ball
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		10/23/07
	Inputs:		MHD mhd,GRID grid
	Outputs:	none
	Purpose:	Performs a Ballistic Relaxation

*/
void ball(MHD *mhd, GRID *grid,int *n,double *t)
{
	double kedx,kedy,kedz,keix,keiy,keiz;
	double tked,tkei,kdo,kio;
	int i,j,k;
	int dustball[4]={10,24,50,100};
	int ionball[4]={10,24,50,100};

	kedx=0.0;
	kedy=0.0;
	kedz=0.0;
	keix=0.0;
	keiy=0.0;
	keiz=0.0;
	for (i=0;i<grid->nx;i++)
	{
		for (j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				kedx=kedx+sqrt(mhd->sx[i][j][k]*mhd->sx[i][j][k])/mhd->rho[i][j][k]/grid->difx[i][j][k]/grid->dify[i][j][k]/grid->difz[i][j][k]/8;
				kedy=kedy+sqrt(mhd->sy[i][j][k]*mhd->sy[i][j][k])/mhd->rho[i][j][k]/grid->difx[i][j][k]/grid->dify[i][j][k]/grid->difz[i][j][k]/8;
				kedz=kedz+sqrt(mhd->sz[i][j][k]*mhd->sz[i][j][k])/mhd->rho[i][j][k]/grid->difx[i][j][k]/grid->dify[i][j][k]/grid->difz[i][j][k]/8;
				keix=keix+sqrt(mhd->six[i][j][k]*mhd->six[i][j][k])/mhd->rhoi[i][j][k]/grid->difx[i][j][k]/grid->dify[i][j][k]/grid->difz[i][j][k]/8;
				keiy=keiy+sqrt(mhd->siy[i][j][k]*mhd->siy[i][j][k])/mhd->rhoi[i][j][k]/grid->difx[i][j][k]/grid->dify[i][j][k]/grid->difz[i][j][k]/8;
				keiz=keiz+sqrt(mhd->siz[i][j][k]*mhd->siz[i][j][k])/mhd->rhoi[i][j][k]/grid->difx[i][j][k]/grid->dify[i][j][k]/grid->difz[i][j][k]/8;
				
			}
		}
	}
	tked=kedx+kedy+kedz;
	tkei=keix+keiy+keiz;
	if (*n == dustball[grid->kdo])
	{
		printf(" ---Applying Ballistic Relaxation (Dust) at n=%5d\n",*n);
		for (i=0;i<grid->nx;i++)
		{
			for (j=0;j<grid->ny;j++)
			{
				for(k=0;k<grid->nz;k++)
				{
					mhd->sx[i][j][k]=0.0;
					mhd->sy[i][j][k]=0.0;
					mhd->sz[i][j][k]=0.0;
				}
			}
		}
		bcorg_nmhdust(mhd,grid,t);
		grid->kdo=grid->kdo+1;
	}
	if (*n == ionball[grid->kio])
	{
		printf(" ---Applying Ballistic Relaxation (Ions) at n=%5d\n",*n);
		for (i=0;i<grid->nx;i++)
		{
			for (j=0;j<grid->ny;j++)
			{
				for(k=0;k<grid->nz;k++)
				{
					mhd->six[i][j][k]=0.0;
					mhd->siy[i][j][k]=0.0;
					mhd->siz[i][j][k]=0.0;
				}
			}
		}
		bcorg_nmhdust(mhd,grid,t);
		grid->kio=grid->kio+1;
	}
}
//*****************************************************************************
/*
	Function:	perturb
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		10/13/10
	Inputs:		MHD mhd,GRID grid
	Outputs:	none
	Purpose:	Performs a Perturbation

*/
void perturb(MHD *mhd, GRID *grid, double *t)
{
        int i,j,k;
	double vin=0.05;
	double vout=1.0;
	double d=1.0;
	double l=10;
        double memi=mhd->me/mhd->mi;
	double memd=mhd->me/mhd->md;

	printf(" --Applying Perturbation!\n");
	for (i=0;i<grid->nx;i++)
	{
		for (j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				//Quantities
				mhd->nuid[i][j][k]=2.16e-5;
				mhd->nuin[i][j][k]=1.83e6;
				mhd->nudn[i][j][k]=7.46;
				mhd->nude[i][j][k]=1.37e-13;
				//Velocities
				mhd->sx[i][j][k]=0.0;
				mhd->sy[i][j][k]=0.0;
				mhd->six[i][j][k]=0.0;
				mhd->siy[i][j][k]=0.0;
				mhd->snx[i][j][k]=0.0;
				mhd->sny[i][j][k]=0.0;
				if (sqrt(grid->y[i][j][k]*grid->y[i][j][k]) < 2*d)
				{
					if (grid->x[i][j][k] > 0)
					{
						mhd->sx[i][j][k]=vout*mhd->rho[i][j][k];
						mhd->six[i][j][k]=vout*mhd->rhoi[i][j][k];
					}
					if (grid->x[i][j][k] < 0)
					{
						mhd->sx[i][j][k]=-vout*mhd->rho[i][j][k];
						mhd->six[i][j][k]=-vout*mhd->rhoi[i][j][k];
					}
				}
				if (sqrt(grid->x[i][j][k]*grid->x[i][j][k]) < l)
				{
					if (grid->y[i][j][k] > 0)
					{
						mhd->sy[i][j][k]=-vin*mhd->rho[i][j][k];
						mhd->siy[i][j][k]=-vin*mhd->rhoi[i][j][k];
					}
					if (grid->y[i][j][k] < 0)
					{
						mhd->sy[i][j][k]=vin*mhd->rho[i][j][k];
						mhd->siy[i][j][k]=vin*mhd->rhoi[i][j][k];
					}
				}
				mhd->sex[i][j][k]=memi*mhd->zi[i][j][k]*mhd->six[i][j][k]-memd*mhd->zd[i][j][k]*mhd->sx[i][j][k]-mhd->me*mhd->jx[i][j][k]/mhd->e;
				mhd->sey[i][j][k]=memi*mhd->zi[i][j][k]*mhd->siy[i][j][k]-memd*mhd->zd[i][j][k]*mhd->sy[i][j][k]-mhd->me*mhd->jy[i][j][k]/mhd->e;
				mhd->sez[i][j][k]=memi*mhd->zi[i][j][k]*mhd->siz[i][j][k]-memd*mhd->zd[i][j][k]*mhd->sz[i][j][k]-mhd->me*mhd->jz[i][j][k]/mhd->e;
			}
		}
	}
	bcorg_nmhdust(mhd,grid,t);
}
//*****************************************************************************
/*
	Function:	energy
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		10/23/07
	Inputs:		MHD mhd,GRID grid,t
	Outputs:	none
	Purpose:	Outputs the ket.txt file

*/
void energy(MHD *mhd, GRID *grid,double t)
{
	FILE *fp;
	double kedx,kedy,kedz,keix,keiy,keiz;
	double kenx,keny,kenz,tken;
	double tked,tkei,kdo,kio;
	double umag,ud,ui,ue,un;
	double difv;
	int i,j,k;

	if (t == grid->dt+grid->dt)
	{
		if((fp=fopen("ket.txt","w")) == NULL)
		{
			printf(" ---ERROR:Couldn't open \"ket.txt\" for output---\n");
			printf("         Check your r/w permissions\n");
			return;
		}
		fprintf(fp,"   Time   |   KEDx   |   KEDy   |   KEDz   |   KEDt   |   KEIx   |   KEIy   |   KEIz   |   KEIt   |   KENx   |   KENy   |   KENz   |   KENt   |   UMAG   |   INTd   |   INTi   |   INTe   |   INTn   |     RHOd     |     RHOi     |     RHOe     |     RHOn     |      Pd      |      Pi      |      Pe      |      Pn      |      Sz      |\n");
	}
	else
	{
		if((fp=fopen("ket.txt","a")) == NULL)
		{
			printf(" ---ERROR:Couldn't open \"ket.txt\" for output---\n");
			printf("         Check your r/w permissions\n");
			return;
		}
	}
//      Calc The Kinetic Energy
	kedx=0.0;
	kedy=0.0;
	kedz=0.0;
	keix=0.0;
	keiy=0.0;
	keiz=0.0;
	kenx=0.0;
	keny=0.0;
	kenz=0.0;
	umag=0.0;
	ud=0.0;
	ui=0.0;
	ue=0.0;
	un=0.0;
	#pragma omp parallel for default(shared) private(i,j,k,difv) reduction(+:kedx,kedy,kedz,keix,keiy,keiz,kenx,keny,kenz,umag,ud,ui,ue,un)
	for (i=0;i<grid->nx;i++)
	{
		for (j=0;j<grid->ny;j++)
		{
			for(k=0;k<grid->nz;k++)
			{
				difv=0.125/(grid->difx[i][j][k]*grid->dify[i][j][k]*grid->difz[i][j][k]);
				kedx=kedx+sqrt(mhd->sx[i][j][k]*mhd->sx[i][j][k])/mhd->rho[i][j][k]*difv;
				kedy=kedy+sqrt(mhd->sy[i][j][k]*mhd->sy[i][j][k])/mhd->rho[i][j][k]*difv;
				kedz=kedz+sqrt(mhd->sz[i][j][k]*mhd->sz[i][j][k])/mhd->rho[i][j][k]*difv;
				keix=keix+sqrt(mhd->six[i][j][k]*mhd->six[i][j][k])/mhd->rhoi[i][j][k]*difv;
				keiy=keiy+sqrt(mhd->siy[i][j][k]*mhd->siy[i][j][k])/mhd->rhoi[i][j][k]*difv;
				keiz=keiz+sqrt(mhd->siz[i][j][k]*mhd->siz[i][j][k])/mhd->rhoi[i][j][k]*difv;
				kenx=kenx+sqrt(mhd->snx[i][j][k]*mhd->snx[i][j][k])/mhd->rhon[i][j][k]*difv;
				keny=keny+sqrt(mhd->sny[i][j][k]*mhd->sny[i][j][k])/mhd->rhon[i][j][k]*difv;
				kenz=kenz+sqrt(mhd->snz[i][j][k]*mhd->snz[i][j][k])/mhd->rhon[i][j][k]*difv;
				umag=umag+0.5*(mhd->bx[i][j][k]*mhd->bx[i][j][k]+mhd->by[i][j][k]*mhd->by[i][j][k]+mhd->bz[i][j][k]*mhd->bz[i][j][k])*difv;
				ud=ud+mhd->p[i][j][k]/mhd->rho[i][j][k]/(mhd->gamma[i][j][k]-1);
				ui=ui+mhd->pi[i][j][k]/mhd->rhoi[i][j][k]/(mhd->gammai[i][j][k]-1);
				ue=ue+mhd->pe[i][j][k]/mhd->rhoe[i][j][k]/(mhd->gammae[i][j][k]-1);
				un=un+mhd->pn[i][j][k]/mhd->rhon[i][j][k]/(mhd->gamman[i][j][k]-1);
				
			}
		}
	}
	tked=kedx+kedy+kedz;
	tkei=keix+keiy+keiz;
	tken=kenx+keny+kenz;
//      Write to ket.txt
	fprintf(fp,"%10.4f %9.4e %9.4e %9.4e %9.4e %9.4e %9.4e %9.4e %9.4e %9.4e %9.4e %9.4e %9.4e %9.4e %9.4e %9.4e %9.4e %9.4e %9.8e %9.8e %9.8e %9.8e %9.8e %9.8e %9.8e %9.8e %9.8e\n",t,kedx,kedy,kedz,tked,keix,keiy,keiz,tkei,kenx,keny,kenz,tken,umag,ud,ui,ue,un,mhd->rho[grid->nx/2][grid->ny/2][grid->nz/2],mhd->rhoi[grid->nx/2][grid->ny/2][grid->nz/2],mhd->rhoe[grid->nx/2][grid->ny/2][grid->nz/2],mhd->rhon[grid->nx/2][grid->ny/2][grid->nz/2],mhd->p[grid->nx/2][grid->ny/2][grid->nz/2],mhd->pi[grid->nx/2][grid->ny/2][grid->nz/2],mhd->pe[grid->nx/2][grid->ny/2][grid->nz/2],mhd->pn[grid->nx/2][grid->ny/2][grid->nz/2],mhd->sz[grid->nx/2][grid->ny/2][grid->nz/2]);
//      Close File	
	fclose(fp);
}

//*****************************************************************************
/*
	Function:	movie
	Author:		Samuel Lazerson (lazersos@gmail.com)
	Date:		10/23/07
	Inputs:		MHD mhd,GRID grid,t
	Outputs:	none
	Purpose:	Outputs the datamovie.dat file

*/
void movie(MHD *mhd, GRID *grid,double t)
{
	FILE *fp;
	int tempd,i,j,k;
	double temp[grid->nx][grid->ny][grid->nz],tempf;
    	double t2d[grid->nx][grid->nz];
    	double ti2d[grid->nx][grid->nz];
    	double te2d[grid->nx][grid->nz];
    	double tn2d[grid->nx][grid->nz];
    	double x3d[grid->nx][grid->ny][grid->nz];
    	double y3d[grid->nx][grid->ny][grid->nz];
    	double z3d[grid->nx][grid->ny][grid->nz];
        // We extract the 2D plots we want for space savings
        for (i=0;i<grid->nx;i++)
        {
            for (k=0;k<grid->nz;k++)
            {
                t2d[i][k]=mhd->rho[i][3][k];
                ti2d[i][k]=mhd->rhoi[i][3][k];
                te2d[i][k]=mhd->rhoe[i][3][k];
                tn2d[i][k]=mhd->rhon[i][3][k];
            }
        }
        
	if (t == 0.0)
	{
		if((fp=fopen("datamovie.dat","wb")) == NULL)
		{
			printf(" ---ERROR:Couldn't open \"datamovie.dat\" for output---\n");
			printf("         Check your r/w permissions\n");
			return;
		}
		// We copy x,y,z to a locally allocated grid for ease of data output.
		for (i=0;i<grid->nx;i++)
		{
			for (j=0;j<grid->ny;j++)
			{
				for(k=0;k<grid->nz;k++)
				{
					x3d[i][j][k]=grid->x[i][j][k];
					y3d[i][j][k]=grid->y[i][j][k];
					z3d[i][j][k]=grid->z[i][j][k];
				}
			}
		}		
		tempd=grid->nx;
		fwrite(&tempd,sizeof(grid->nx),1,fp);
		tempd=grid->ny;
		fwrite(&tempd,sizeof(grid->ny),1,fp);
		tempd=grid->nz;
		fwrite(&tempd,sizeof(grid->nz),1,fp);
		fwrite(x3d,sizeof(x3d),1,fp);
		fwrite(y3d,sizeof(y3d),1,fp);
		fwrite(z3d,sizeof(z3d),1,fp);
		fwrite(&t,sizeof(t),1,fp);
		fwrite(t2d,sizeof(t2d),1,fp);
		fwrite(ti2d,sizeof(ti2d),1,fp);
		fwrite(te2d,sizeof(te2d),1,fp);
		fwrite(tn2d,sizeof(tn2d),1,fp);

	}
	else
	{
		if((fp=fopen("datamovie.dat","ab")) == NULL)
		{
			printf(" ---ERROR:Couldn't open \"datamovie.dat\" for output---\n");
			printf("         Check your r/w permissions\n");
			return;
		}
		fwrite(&t,sizeof(t),1,fp);
		fwrite(t2d,sizeof(t2d),1,fp);
		fwrite(ti2d,sizeof(ti2d),1,fp);
		fwrite(te2d,sizeof(te2d),1,fp);
		fwrite(tn2d,sizeof(tn2d),1,fp);
	}
//      Close File	
	fclose(fp);
}

/*
	Function:	handle_error
	Written by:	http://www.unidata.ucar.edu/software/netcdf/docs/netcdf-c/nc_005fstrerror.html#nc_005fstrerror
	Date:		Unknown
	Purpose:	To handle netCDF errors
*/
void handle_error(int status) 
{
	if (status != NC_NOERR) 
	{
        	fprintf(stderr, "%s\n", nc_strerror(status));
        }
}

/*
	Function:	allocate_grid
	Written by:	Samuel Lazerson
	Date:		10/22/10
	Purpose:	To allocate the elements of the structures
*/
void    allocate_grid(MHD *mhd, GRID *grid)
{
	// Allocate MHD
	mhd->rho = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->rhoi = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->rhoe = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->rhon = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->sx = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->sy = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->sz = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->six = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->siy = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->siz = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->sex = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->sey = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->sez = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->snx = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->sny = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->snz = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->bx = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->by = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->bz = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->jx = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->jy = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->jz = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->ex = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->ey = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->ez = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->p = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->pi = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->pe = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->pn = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->zd = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->zi = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->gamma = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->gammai = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->gammae = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->gamman = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->nuid = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->nuie = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->nude = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->nudn = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->nuin = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->nuen = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->gravx = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->gravy = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->gravz = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	mhd->rhosmo = (unsigned short int ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(unsigned short int));
	mhd->rhoismo = (unsigned short int ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(unsigned short int));
	mhd->rhonsmo = (unsigned short int ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(unsigned short int));
	mhd->sxsmo = (unsigned short int ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(unsigned short int));
	mhd->sysmo = (unsigned short int ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(unsigned short int));
	mhd->szsmo = (unsigned short int ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(unsigned short int));
	mhd->sixsmo = (unsigned short int ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(unsigned short int));
	mhd->siysmo = (unsigned short int ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(unsigned short int));
	mhd->sizsmo = (unsigned short int ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(unsigned short int));
	mhd->snxsmo = (unsigned short int ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(unsigned short int));
	mhd->snysmo = (unsigned short int ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(unsigned short int));
	mhd->snzsmo = (unsigned short int ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(unsigned short int));
	mhd->psmo = (unsigned short int ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(unsigned short int));
	mhd->pismo = (unsigned short int ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(unsigned short int));
	mhd->pesmo = (unsigned short int ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(unsigned short int));
	mhd->pnsmo = (unsigned short int ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(unsigned short int));
	// Allocate GRID
	grid->x = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->y = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->z = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->difx = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->dify = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->difz = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->ddifx = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->ddify = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->ddifz = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->ddifpx = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->ddifpy = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->ddifpz = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->ddifmx = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->ddifmy = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->ddifmz = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->meanpx = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->meanpy = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->meanpz = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->meanmx = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->meanmy = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->meanmz = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->ssx    = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->ssy    = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->ssz    = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->s      = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->bx0    = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->by0    = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
	grid->bz0    = (double ***) newarray(grid->nx,grid->ny,grid->nz,sizeof(double));
}

/*
	Function:	newarray
	Written by:	
	Date:		Unknown
	Purpose:	To allocate 3D arrays of size icount, jcount, kcount
	Adapted from:
	http://stackoverflow.com/questions/2438142/dynamic-memory-allocation-for-3d-array/2438624#2438624
*/
void*** newarray(int icount, int jcount, int kcount, int type_size)
{
	int i,j,k;
	void*** iret = (void***)malloc(icount*sizeof(void***)+icount*jcount*sizeof(void**)+icount*jcount*kcount*type_size);
	void** jret = (void**)(iret+icount);
	char* kret = (char*)(jret+icount*jcount);
	for(i=0;i<icount;i++)
	{
		iret[i] = &jret[i*jcount];
	}
    	for(i=0;i<icount;i++)
        {
		for(j=0;j<jcount;j++)
		{
			jret[i*jcount+j] = &kret[(i*jcount+j)*kcount*type_size];
		}
	}
	return iret;
}

/*
	Function:	min3d
	Written by:	Samuel Lazerson (lazersos@gmail.com)
	Date:		11/16/10
	Purpose:	Finds the minimum of a 3D array
*/
double min3d(double ***array,int nx, int ny, int nz)
{
	int i,j,k;
	double val;
	
	val=array[0][0][0];
	for (i=0;i<nx;i++)
	{
		for (j=0;j<ny;j++)
		{
			for(k=0;k<nz;k++)
			{
				if (array[i][j][k] < val) val=array[i][j][k];				
			}
		}
	}
	return(val);
}

/*
	Function:	max3d
	Written by:	Samuel Lazerson (lazersos@gmail.com)
	Date:		11/16/10
	Purpose:	Finds the maximum of a 3D array
*/
double max3d(double ***array,int nx, int ny, int nz)
{
	int i,j,k;
	double val;
	
	val=array[0][0][0];
	for (i=0;i<nx;i++)
	{
		for (j=0;j<ny;j++)
		{
			for(k=0;k<nz;k++)
			{
				if (array[i][j][k] > val) val=array[i][j][k];				
			}
		}
	}
	return(val);
}


C*************************************************************************************
C     THIS CODE GENERATES THE POSITIONS OF WATER MOLECULES (RANDOMLY) IN A CUBIC     *
C     PERIODIC CELL. CONSTRAINTS ARE IMPOSED ON  THE MINIMUM OXYGEN-OXYGEN           *
C     SEPARATIONS (SEE THE DESCRIPTION INSIDE THE CODE).                             *
C                                                                                    *
C     AUTHOR: DR. RAYMOND ATTA-FYNN                                                  *
C     AFFILIATION: EMSL, PACIFIC NORTHWEST NATIONAL LAB, BOX 999, RICHLAND WA 99352  *
C                                                                                    *
C     SEPTEMBER 20, 2009                                                             *
C                                                                                    *
C     KINDLY CITE MY PAPER IF YOU USE THIS CODE TO GENERATE INITIAL STRUCTURES       *
C     FOR YOUR SIMULATION                                                            *                          
C                                                                                    * 
C                                                                                    *
C     DEFINITION OF VARIABLES                                                        *
C     _______________________                                                        *
C                                                                                    *
C                                                                                    *
C     ATOM_SYMB     : ATOMIC SYMBOLS                                                 *
C     H_MASS        : ATOMIC MASS OF HYDROGEN                                        *
C     O_MASS        : ATOMIC MASS OF OXYGEN                                          *
C     H2O_MASS      : MOLAR MASS OF WATER                                            *
C     AVOGADRO      : AVOGADRO'S NUMBER                                              *
C     RHO           : DENSITY OF WATER (IN GRAMS/CM^3)                               *
C     BOX_L         : BOX LENGTH (IN ANGSTROMS)                                      *
C     NWM           : TOTAL NUMBER OF WATER MOLECULES                                *
C     NAT           : TOTAL NUMBER OF ATOMS=3*NWM                                    *
C     POS(3,NMAX)   : ATOMIC POSITION (IN ANGSTROMS)                                 *
C     D_OH          : O-H BOND DISTANCE (IN ANGSTROMS)                               *
C     ANG_HOH       : H-O-H BOND ANGLE  (IN DEGREES)                                 *
C     ANG_EPS       : H-O-H BOND ANGLE  TOLERANCE(IN DEGREES)                        *
C     RCUT_OO       : MIMIMUM O-O  SEPARATION (IN ANGSTROM)                          *
C     RAN2          : RANDOM NUMBER GENERATOR FUNCTION (FROM NUMERICAL RECIPES)      *
C     ISEED         : RANDOM NUMBER SEED                                             *
C     DIST(A,B,L)   : FUNCTION WHICH COMPUTES THE DISTANCE BETWEEN POINTS AND B      *  
C                     IN BOX OF LENGTH L SUBJECT TO PERIODIC BOUNDARY CONDITIONS     *
C     ICOUNT        : ATOM COUNTER                                                   *
C     OUTPUT_FILE   : NAME OF FILE IN WHICH OUTPUT IS WRITTEN                        *
C                                                                                    *
C*************************************************************************************


      PROGRAM                    BUILD_WATER
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER                 (N1=10000)
C     N1 IS THE MAXIMUM NUMBER OF WATER MOLECULES
      CHARACTER                  ATOM_SYMBOL(3*N1)*2, OUTPUT_FILE*30
      DIMENSION                  POS(3,3*N1), ATOM_MASS(3*N1) 
      DIMENSION                  TMP0(3), TMP1(3)
      EXTERNAL                   RAN2, DISTANCE
      DATA                       ISEED/-1/, PI/3.14159265358979D+0/
      DATA                       AVOGADRO/6.02214179D+23/


C*************************************************************************************
C              EDIT THIS SECTION ACCORDING TO YOUR NEEDS                             *
C*************************************************************************************

C     (A) FIXED INPUT PARAMETERS
C     NOTE: TO SIMULATE DEUTERIUM SET H_MASS = 2.01410178D0 

      H_MASS      =  1.008D0        
      O_MASS      =  15.9994D0
      H2O_MASS    =  O_MASS + 2.D0*H_MASS
 
C     (B) VARIABLE INPUT PARAMETERS: 
C         SET THE NUMBER OF WATER MOLECULES NWM, 
C         SET THE DENSITY OF WATER RHO IN G/CM**3
C         SET THE O--H BOND LENGTH D_OH IN ANGSTROM
C         SET THE H-O-H ANGLE ANG_HOH IN DEGREES
C         SET THE ANGLE TOLERANCE ANG_EPS IN DEGREES
C         SET THE MINIMUM O--O BOND LENGTH RCUT_OO IN ANGSTROM
C         SET THE NAME OF THE OUTPUT FILE

      NWM         =   4
      RHO         =   0.997D0
      D_OH        =   1.0D0                  
      ANG_HOH     =   109.47D0
      ANG_EPS     =   1.D-5             
      RCUT_OO     =   2.7D0
      OUTPUT_FILE =  'h2o-12.xyz'

C*************************************************************************************
C              EDITABLE SECTION ENDS HERE                                            *
C*************************************************************************************




C*************************************************************************************
C              COMPUTING THE LENGTH OF THE BOX                                       *
C*************************************************************************************

C     DETERMINE THE TOTAL MASS
      TOTAL_MASS = DFLOAT(NWM)*H2O_MASS

C     COMPUTE THE BOX LENGTH IN ANGSTROM
      BOX_L =  1.D8*((TOTAL_MASS/(AVOGADRO*RHO))**(1.D0/3.D0))


C*************************************************************************************
C              SUCCESSIVE GENERATION OF WATER MOLECULES                              *
C*************************************************************************************

C     IN THIS SECTION OF THE CODE, WATER MOLECULES WILL BE RANDOMLY GENERATED IN A  
C     CUBIC BOX WITH PREDIODIC BOUNDARY CONDITIONS AND FOLlOWING CONSTRAINTS: 
C     THE O-O DISTANCE IS AT LEAST RCUT_OO.

C     IMPORTANT NOTES:
C     (a) ANY FAIRLY GOOD RANDOM NUMBER GENERATOR IS OKAY. THE RANDOM NUMBER USED 
C         HERE WAS TAKEN VERBATIM FROM NUMERICAL RECIPES (F77 VERSION)
C
C     (b) IN THIS CODE NO CONSTRAINT WAS IMPOSED ON THE ORIENTATION OF THE WATER 
C         DIPOLE VECTORS; THEY ARE RANDOM. 


C     NAT IS THE TOTAL NUMBER OF ATOMS      
      NAT    =  3*NWM
      ICOUNT = 0

      DO WHILE (ICOUNT.LT.NAT)

C        RANDOMLY GENERATE AN OXYGEN ATOMIC POSITION IN THE INTERVAL [-L/2, L/2]^3
         DO 11 I = 1, 3
            TMP0(I) = BOX_L*(RAN2(ISEED)-0.5D0)
 11      CONTINUE
     
         IF(ICOUNT.EQ.0)GOTO 16

C        LOOP OVER ALL PREVIOUSLY GENERATED O-ATOMS UP TILL NOW          
         DO 14 I = 1, ICOUNT

C           MAKE SURE H ATOMS ARE NOT USED IN THE CONSTRAINT CHECK
            IF(MOD(I+2,3).NE.0)GOTO 14   

C           TEMPORARILY STORE THE O POSITION IN TMP1
            DO 15 J = 1, 3
               TMP1(J) = POS(J,I) 
 15         CONTINUE

C           CHECK IF O-O DISTANCE CONSTRAINT IS SATISFIED, IF NOT REJECT
            DOO = DISTANCE(TMP0,TMP1,BOX_L) 
            IF(DOO.LT.RCUT_OO)GOTO 27

 14      CONTINUE

 16      CONTINUE 

C        ACCEPT THE GENERATED OXYGEN ATOM
         ICOUNT = ICOUNT + 1
         DO 17 I = 1, 3
            POS(I,ICOUNT) = TMP0(I)
 17      CONTINUE

C        RECORD THE ATOMIC SYMBOL AND MASS
         ATOM_SYMBOL(ICOUNT) = 'O'
         ATOM_MASS(ICOUNT) = O_MASS    

C****************************FIRST H POSITION WILL BE GENERATED HERE****************** 

C       THE PROCEDURE IS STRAIGTH-FOWARD: SIMPLY GENERATE A RANDOM VECTOR OF 
C       LENGHT D_OH. WITH O AS THE ORIGIN OF THE VECTOR, SIMPLY ADD H TO THE 
C       OTHER END

 18     CONTINUE

C       RANDOM GENERATE A VECTOR ALONG WHICH THE OH BOND WILL BE FORMED. 
        V_MODULUS = 0.D0
         DO 19 I = 1, 3
            TMP0(I) = 2.D0*RAN2(ISEED)-1.D0
            V_MODULUS = V_MODULUS + TMP0(I)*TMP0(I)
 19      CONTINUE
         V_MODULUS = DSQRT(V_MODULUS)

C        ENSURE THAT THE MODULUS OF THE VECTOR IS NOT CLOSE TO ZERO
         IF(V_MODULUS.LT.1.D-5)GOTO 18

C        MAKE THE LENGTH OF THE VECTOR 1
         DO 20 I = 1, 3
            TMP0(I) = TMP0(I)/V_MODULUS
 20      CONTINUE

C        GENERATE THE FIRST H ATOMIC POSITION USING TMP0
         ICOUNT = ICOUNT + 1
         DO 21 I = 1, 3
            POS(I,ICOUNT) = POS(I,ICOUNT-1)+D_OH*TMP0(I)
 21      CONTINUE

C        RECORD THE ATOMIC SYMBOL AND MASS
         ATOM_SYMBOL(ICOUNT) = 'H'
         ATOM_MASS(ICOUNT) = H_MASS     

C****************************GENERATION OF FIRST H POSITION DONE********************** 



C****************************SECOND H POSITION WILL BE GENERATED HERE*****************

C       THE PROCEDURE TO GENERATE THE SECOND-HYDROGEN IS SIMILAR TO THE FIRST
C       HOWEVER, THE CONSTRAINT WATER BOND ANGLE CONSTRAINT MUST BE SATISFIED.


C       RANDOM GENERATE A VECTOR ALONG WHICH THE OH BOND WILL BE FORMED. 
 22      CONTINUE
         V_MODULUS = 0.D0
         DO 23 I = 1, 3
            TMP0(I) = 2.D0*RAN2(ISEED)-1.D0
            V_MODULUS = V_MODULUS + TMP0(I)*TMP0(I)
 23      CONTINUE 
         V_MODULUS = DSQRT(V_MODULUS)

C        ENSURE THAT THE MODULUS OF THE VECTOR IS NOT CLOSE TO ZERO
         IF (V_MODULUS.LT.1.D-5)GOTO 22

C        MAKE THE LENGTH OF THE VECTOR 1
         DO 24 I = 1, 3
            TMP0(I) = TMP0(I)/V_MODULUS
 24      CONTINUE

C        TEMPORARILIY GENERATE THE SECOND H POSTION (STORED IN TMP0)
C        ALSO STORE THE FIRST H POSITION (GENERATED IN THE PREVIOUS BLOCK) IN TMP1
         DO 25 I = 1, 3
            TMP0(I) = POS(I,ICOUNT-1)+D_OH*TMP0(I)
            TMP1(I) = POS(I,ICOUNT)
 25      CONTINUE

C        COMPUTE THE H-O-H BOND ANGLE USING THE LAW OF COSINES
         DHH = DISTANCE(TMP0,TMP1,BOX_L)
         ANG_TMP = DACOS((D_OH**2 + D_OH**2 - DHH**2)/(2.D0*D_OH**2))
         ANG_TMP = 180.D0*ANG_TMP/PI

C        CHECK IF BOND ANGLE IS WITHIN TOLERANCE, IF NO REJECT
         ANG_TMP = ANG_TMP-ANG_HOH
         IF(DABS(ANG_TMP).GT.ANG_EPS)GOTO 22
         
C        GENERATE THE SECOND H ATOMIC POSITION
         ICOUNT = ICOUNT + 1
         DO 26 I = 1, 3
            POS(I,ICOUNT) = TMP0(I)
 26      CONTINUE

C        RECORD THE ATOMIC SYMBOL AND MASS
         ATOM_SYMBOL(ICOUNT) = 'H'
         ATOM_MASS(ICOUNT) = H_MASS     

C        PRINT SUCCESSFUL GENERATION WATER MOLECULE ON THE SCREEN
         WRITE(*,32)ICOUNT/3

 27      CONTINUE

      END DO

C*************************************************************************************
C                       GENERATION OF WATER MOLECULES DONE                           *
C*************************************************************************************


C*************************************************************************************
C                       PRINTING OUTPUT                                              *
C*************************************************************************************

      OPEN(28,FILE=OUTPUT_FILE,STATUS='UNKNOWN')
      WRITE(28,'(I7)')NAT
      WRITE(28,'(F12.4)')BOX_L

C     SHIFT ALL ATOMS TO [-L/2,L/2]^3

      BOX_L_INV = 1.D0/BOX_L
     
      DO 29 I = 1, NAT
         DO 30 J = 1, 3
            TMP = POS(J,I)
            TMP = TMP - BOX_L*DNINT(TMP*BOX_L_INV )
            POS(J,I)=TMP
 30      CONTINUE
         WRITE(28,33)ATOM_SYMBOL(I),(POS(J,I),J=1,3),ATOM_MASS(I)
 29   CONTINUE
      CLOSE(28)
    

C*************************************************************************************
C                       OUTPUT PRINTING DONE                                         *
C*************************************************************************************



      WRITE(*,34)OUTPUT_FILE

 32   FORMAT('GENERATED WATER MOLECULE',2X,I7)
 33   FORMAT(A,4F12.6)
 34   FORMAT(//,'TASK COMPLETED',//
     >       'OUTPUT XYZ FILE IS:',2X,A,//)


      STOP

      END


C     END OF PROAGRAM



C*************************************************************************************
C                       DISTANCE FUNCTION                                            *
C*************************************************************************************

      FUNCTION                   DISTANCE(A, B, BOX_L)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION                  A(3), B(3)

      BOX_L_INV = 1.D0/BOX_L
      DISTANCE  = 0.D0

      DO 10 I = 1, 3
         TMP = A(I)-B(I)
         TMP = TMP - BOX_L*DNINT(TMP*BOX_L_INV)
         DISTANCE = DISTANCE + TMP**2
 10   CONTINUE

      DISTANCE = DSQRT(DISTANCE)

      RETURN

      END
      

C*************************************************************************************
C                       DISTANCE FUNCTION ENDS HERE                                  *
C*************************************************************************************




C*************************************************************************************
C            FORTRAN 77 RANDOM NUMBER GENERATOR                                      *
C REFERENCE: NUMERICAL RECIPES IN FORTRAN 77: THE ART OF SCIENTIFIC COMPUTING        *
C            SECOND EDITION (CAMBRIDGE UNIVERSITY PRESS, 1992)                       *
C*************************************************************************************

C     Long period (> 2 x 10^18) random number generator of L'Ecuyer 
C     with Bays-Durham shuffle and added safeguards. Returns a uniform 
C     random deviate between 0.0 and 1.0 (exclusive of the endpoint values). 
C     Call with IDUM a negative integer to initialize; thereafter, do not
C     alter IDUM between successive deviates in a sequence. RNMX should 
C     approximate the largest floating value that is less than 1.

      FUNCTION RAN2(IDUM)
      INTEGER IDUM,IM1,IM2,IMM1,IA1,IA2,IQ1,IQ2,IR1,IR2,NTAB,NDIV
      DOUBLE PRECISION ran2,AM,EPS,RNMX
      PARAMETER (IM1=2147483563,IM2=2147483399,AM=1.D0/IM1,
     >IMM1=IM1-1,IA1=40014,IA2=40692,IQ1=53668,IQ2=52774,IR1=12211,
     >IR2=3791,NTAB=32,NDIV=1+IMM1/NTAB,EPS=1.2D-7,RNMX=1.D0-EPS)
      INTEGER IDUM2,J,K,IV(NTAB),IY
      SAVE IV,IY,IDUM2
      DATA IDUM2/123456789/, IV/NTAB*0/, IY/0/

      IF(IDUM.LT.0)THEN                !Initialize.

         IDUM=MAX(-IDUM,1)             !Be sure to prevent IDUM = 0.
         IDUM2=IDUM
         DO 11 J=NTAB+8,1,-1           !Load the shuffle table (after 8 warm-ups).
           IDUM=IA1*(IDUM-K*IQ1)-K*IR1
           IF(IDUM.LT.0) IDUM=IDUM+IM1
           IF (J.LE.NTAB) IV(J)=IDUM
 11      CONTINUE
         IY=IV(1)

      END IF

      K=IDUM/IQ1                       !Start here when not initializing.
      IDUM=IA1*(IDUM-K*IQ1)-K*IR1      !Compute IDUM=mod(IA1*IDUM,IM1) without over

      IF(IDUM.LT.0) IDUM=IDUM+IM1      !flows by Schrage's method.
      K=IDUM2/IQ2
      IDUM2=IA2*(IDUM2-K*IQ2)-K*IR2    !Compute IDUM2=mod(IA2*IDUM2,IM2) likewise.

      IF(IDUM2.LT.0)IDUM2=IDUM2+IM2
      J=1+IY/NDIV                      !Will be in the range 1:NTAB.
      IY=IV(J)-IDUM2                   !Here IDUM is shuffled, IDUM and IDUM2 are com-
      IV(J)=IDUM                       !bined to generate output.

      IF(IY.LT.1)IY=IY+IMM1
      RAN2=MIN(AM*DFLOAT(IY),RNMX)     !Because users don't expect endpoint values.
      RETURN
      END
          

C*************************************************************************************
C                       RANDOM NUMBER GENERATOR ROUTINE ENDS HERE                    *
C*************************************************************************************

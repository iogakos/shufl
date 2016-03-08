LATEX=latex

DVIPS=dvips
DVIPSFLAGS= -Ppdf 

PSPDF=ps2pdf
PDFFLAGS= -dPDFSETTINGS=/prepress

PSNUP=psnup
PSNUPFLAGS= -w85cm -h119cm -f 

poster.pdf : poster.ps poster.dvi 
	${PSPDF} ${PDFFLAGS} poster.ps 

poster.ps : poster.dvi
	${DVIPS} ${DVIPSFLAGS} -o poster.ps poster.dvi 

poster.dvi : poster.tex
	${LATEX} poster

clean :
	\rm *.dvi
	\rm *.aux
	\rm *.log
	\rm *.out


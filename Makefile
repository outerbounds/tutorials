copy-intro-to-mf:
	rsync -a --include 'season-*/' --include 'intro-to-mf/' --include '*.py' --exclude '*' ../docs/docs/tutorials/nbs/intro-to-mf .	

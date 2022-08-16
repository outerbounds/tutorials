sync:
	rm intro-to-mf/season-*/*.py
	rsync -a --include 'season-*/' --include 'intro-to-mf/' --include '*.py' --exclude '*' ../docs/docs/tutorials/nbs/intro-to-mf .	

s1-scripts:
	cd ./intro-to-mf/season-1 && python minimum_flow.py run && python decorator_flow.py run && python artifact_flow.py run && python parameter_flow.py run

s2-scripts:
	cd ./intro-to-mf/season-2 && python random_forest_flow.py run && python gradient_boosted_trees_flow.py run && python branching_trees_flow.py run

s3-scripts:
	cd ./intro-to-mf/season-3 && python neural_net_flow.py run --e 2 && python neural_net_card_flow.py run --e 2 && python debuggable_flow.py run	

all-scripts:
	make s1-scripts
	make s2-scripts
	make s3-scripts 

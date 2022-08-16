sync:
	rm intro-to-mf/season-*/*.py
	rsync -a --include 'season-*/' --include 'intro-to-mf/' --include '*.py' --exclude '*' ../docs/docs/tutorials/nbs/intro-to-mf .	

all:
	python intro-to-mf/season-1/minimum_flow.py run
	python intro-to-mf/season-1/decorator_flow.py run
	python intro-to-mf/season-1/artifact_flow.py run
	python intro-to-mf/season-1/parameter_flow.py run
	python intro-to-mf/season-2/random_forest_flow.py run
	python intro-to-mf/season-2/gradient_boosted_trees_flow.py run
	python intro-to-mf/season-2/branching_trees_flow.py run
	python intro-to-mf/season-3/neural_net_flow.py run
	python intro-to-mf/season-3/neural_net_card_flow.py run
	python intro-to-mf/season-3/debuggable_flow.py run	

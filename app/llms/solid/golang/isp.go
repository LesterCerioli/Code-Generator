package main

type Worker interface {
	Work()
	Eat()
}

type Robot struct{}

func (r Robot) Work() {}
func (r Robot) Eat()  {}

type Workable interface {
	Work()
}

type Eatable interface {
	Eat()
}

package main

type Bird interface {
	Fly()
}

type Ostrich struct{}

func (o Ostrich) Fly() {
	panic("Ostrich cannot fly")
}

type FlyingBird interface {
	Fly()
}

type Sparrow struct{}

func (s Sparrow) Fly() {}

package main

func CalculateDiscount(t string, price float64) float64 {
	if t == "student" {
		return price * 0.9
	}
	return price
}

type Discount interface {
	Apply(price float64) float64
}

type StudentDiscount struct{}

func (d StudentDiscount) Apply(price float64) float64 {
	return price * 0.9
}

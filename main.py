from tokenizer.tokenizer import Tokenizer, print_token_stream

if __name__ == '__main__':
    _t = Tokenizer('123a_abc+764 =- obj/62 *+9 az++ or s=90-7 and q+=-8 or w/=98-i\n'
                   ' bc*=8 or ee+q  -=4gt9 % +abc xor u%=99-3*q/d')
    _t.tokenize()
    print_token_stream(_t.tokens, True)

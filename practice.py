import numpy as np
import bisect
import itertools
from random import sample 
print("/home/data/project/hash")

class Hash():
    def __init__(self, filename):
        # attributes
        self.filename = filename # "/home/data/project/hash/b_small.in"
        self.input_list = []
        self.people = 0
        self.num_pizza = 0

        self.small_cnt = 20
        self.target = 90000 # target for large sum 이걸 넘어가도록 큰 숫자 정하기!
        
        #self.mean = np.average(self.input_list) 에러남

        self.small_scores = []
        self.sum2choice = dict() # 작은 숫자 합을 만들기 위한 인덱스 조합 찾는용

        self.max_score = 0
        self.max_choice = None

        # run !!
        self.people, self.pizza, self.input_list = self.read_input_file(self.filename)
        self.make_target() # 작은거 더해서 만들수 있는 최대치만큼 빼고 타겟 설정
        self.make_small_scores()
        



    def make_target(self):
        s = sum(self.input_list[:self.small_cnt])
        if s > self.people:
            self.target = 0
        else:
            self.target = self.people - s


    def large1(self):
        # 큰거부터 더해서 타겟 근처로 가보기
        s = 0
        select = []
        for i, elem in enumerate(reversed(self.input_list)):
            idx = self.num_pizza - 1 - i 
            if s + elem < self.target:
                s += elem
                select.append(idx)
            else:
                break
        return s, select

    def mideum_somangsarang(self):
        idle_num = int(self.people / self.mean) # 전체 평균이 있다는 가정에서
        start_idx = int((self.num_pizza - idle_num)/2)
        idle_sum = sum(self.input_list[start_idx:start_idx+idle_num])
        sub_target = self.people - idle_sum

        # 중앙값 이동
        if sub_target > 0:
            diff = self.input_list[start_idx+idle_num] - self.input_list[start_idx]
            if diff < idle_num:
                pass
            else:
                while diff < idle_num:
                    diff = self.input_list[start_idx+idle_num] - self.input_list[start_idx]
                    idle_sum += diff
                    start_idx += 1
        else:
            diff = self.input_list[start_idx+idle_num-1] - self.input_list[start_idx-1]
            if diff < idle_num:
                pass
            else:
                while diff < idle_num:
                    diff = abs(self.input_list[start_idx+idle_num] - self.input_list[start_idx])
                    idle_sum -= diff
                    start_idx -= 1

        selected = []
        for i in idel_num:
            selected.append(start_idx+i)

        return idle_sum, selected

    def read_input_file(self, filename):
        with open(filename) as f:
            m, n = f.readline().split()
            m = int(m.strip()) # 이게 총 인원수
            #n = int(n.strip()) # --> 이거 피자 타입 개수라 무의미 할듯... len(size_list)
            size_list = f.readline().split()
            size_list = [int(s.strip()) for s in size_list if int(s.strip()) <= m]
        return m, len(size_list), size_list

    def make_small_scores(self):
        # 작은 점수 조합 리스트 만들기 (바이너리 서치용)
        self.small_scores = []
        self.sum2choice = dict()
        max_cnt = min(self.small_cnt, self.pizza)
        indices = list(range(max_cnt))
        for r in range(1, max_cnt+1):
            for tup in itertools.combinations(indices, r):
                s = sum([self.input_list[i] for i in tup])
                if s <=self.people and s not in self.sum2choice:
                    self.small_scores.append(s)
                    self.sum2choice[s] = tup
                    if self.max_score < s:
                        self.max_score = s
                        self.max_choice = tup
        self.small_scores.sort()
        #print("make 318", self.sum2choice[318])


    def find_score(self, large_sum, large_choice):
        # large_sum = 큰 수 합
        # large_choice = 고른 idx 들
        target = self.people - large_sum
        idx = bisect.bisect_right(target, self.small_scores)

        small = self.small_scores[idx-1]
        small_choice = self.sum2choice[small]
        final_choice = set(large_choice)
        for i in small_choice:
            if i in final_choice:
                raise Exception("Large choice contains small choice")
            else:
                final_choice.add(i)
        return large_sum + small, final_choice

    def get_small_group_offset(self, sum_thres=10000):
        # 앞에서 부터 총 합이 sum_thres를 넘기 직전의 인덱스를 구합니다
        total = 0
        for i, n in enumerate(self.input_list):
            total += n
            if total > sum_thres:
                break
        return i - 1
    
    def make_random_sampled_sum_list(self, input_list, min_idx=None, max_idx=None, times=1):
        # 인풋 리스트의 합의 조합을 (적당히 많이) 만듭니다 ^^
        # 조합은 너무 오래 걸리므로, 개수(k)별로 k/len(input_list)번 반복 랜덤 샘플링 후 합산
        # 조합을 n배 하고 싶다면 times 파라미터를 변경
        # min ~ max (inclusive)
        if min_idx and max_idx:
            input_list = input_list[min_idx:max_idx+1]
        elif min_idx:
            input_list = input_list[min_idx:]
        elif max_idx:
            input_list = input_list[:max_idx+1]
        
        random_combi_sum_set = set()
        for k in range(1, len(input_list)):
            repeat = int(len(input_list)/k) * times
            for i in range(repeat):
                random_sample = sample(input_list, k)
                random_combi_sum_set.add(sum(random_sample))
                
        return sorted(list(random_combi_sum_set))

    def find_score_by_random_combi_sum(self, ratio_for_sum_thres=0.1):
		# 일단 전체 인원수 대비 함수 파라미터 ratio 만큼의 값을 구함
		# 스몰 그룹의 총합이 그 만큼을 못 넘도록 스몰 그룹을 구함
        small_group_sum_thres = int(self.people * ratio_for_sum_thres)
        offset = self.get_small_group_offset(sum_thres=small_group_sum_thres)

        small_group = self.input_list[:offset]
        big_group = self.input_list[offset:]
       	
		# 각각에서 랜덤 샘플 방식으로 적당히 조합을 구함
        small_group_sm_combi = self.make_random_sampled_sum_list(small_group)
        big_group_sum_combi = self.make_random_sampled_sum_list(big_group)
		
		# 이미 정렬되어 있기 때문에 거꾸로 보면서 만족하는 합이 최적값
        max_num_slice = -1
        for b in reversed(big_group_sum_combi):
            for s in reversed(small_group_sm_combi):
                total_slice = s + b
                if total_slice < self.people:
                     return total_slice
        return 0 
        
    def bestAlgorithms(self):
        pass
    


if __name__ == "__main__":
    # h = Hash("./b_small.in")
    # h = Hash("./c_medium.in")
    # h = Hash("./d_quite_big.in")
    h = Hash("./e_also_big.in")

    #mid_sum, selected = h.mideum_somangsarang()
    #print(mid_sum, selected)

    print(h.find_score_by_random_combi_sum())


'''
1. 작 
2. 큰(펭수)
- 아이디어1
    - 
3. 바써
4. 어큰
5. 다방
'''

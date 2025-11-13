import cpmpy
from cpmpy import *
from cpmpy.expressions.core import Operator, Comparison
from cpmpy.transformations.negation import push_down_negation
from cpmpy.transformations.normalize import toplevel_list
from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView

from utils.utils import extract_dv_from_expression
from utils.utils_classes import Relation
from utils.utils_lgp import define_dict_adjacency


class LGProblem():

    def __init__(self, type):
        self.type = type


    def make_model(self):
        constraints = []
        facts = []
        dict_constraint_type = {}
        dict_constraint_dv = {}
        dict_constraints_clues = {}
        explainable_facts = set()
        if self.type == 0:
            month = ["1", "2", "3", "4", "5"]
            home = ["the_other_home", "hughenden", "wolfenden", "markmanor", "barnhill"]
            address = ["the_other_type1", "circle_drive", "bird_road", "grant_place", "fifth_avenue"]
            ghost = ["the_other_type2", "victor", "lady_grey", "brunhilde", "abigail"]

            types = [month, home, address, ghost]
            n = len(types)
            m = len(types[0])
            assert all(len(types[i]) == m for i in range(n)), "all types should have equal length"

            # relations between types
            investigated_in = Relation(home, month, name='investigated')  # investigated_in(home, month)
            situated = Relation(home, address, name='situated')  # on(home, type1)
            haunted_by = Relation(home, ghost, name='haunted')  # haunted_by(home, type2)
            is_linked_with_1 = Relation(month, address, name='first')  # is_linked_with_1(month, type1)
            is_linked_with_2 = Relation(month, ghost, name='second')  # is_linked_with_2(month, type2)
            is_linked_with_3 = Relation(address, ghost, name='third')  # is_linked_with_3(type1, type2)

            rels = [investigated_in, situated, haunted_by, is_linked_with_1, is_linked_with_2, is_linked_with_3]
            rels_visualization = [[haunted_by.df, situated.df, investigated_in.df],
                                  [is_linked_with_2.df, is_linked_with_1.df],
                                  [is_linked_with_3.df]]

            # Bijectivity
            constraints_row,constraints_columns = self.buildBijectivity(rels)
            for cnstr in constraints_row:
                constraints.append(cnstr)
                dict_constraints_clues[cnstr] = 'bijectivity'
                dict_constraint_dv[cnstr] = extract_dv_from_expression(cnstr,involved=[])
                dict_constraint_type[cnstr] = 'bijectivity'
            for cnstr in constraints_columns:
                constraints.append(cnstr)
                dict_constraints_clues[cnstr] = 'bijectivity'
                dict_constraint_dv[cnstr] = extract_dv_from_expression(cnstr,involved=[])
                dict_constraint_type[cnstr] = 'bijectivity'


            # Transitivity
            constraints_transitivity = [[] for _ in range(12)]
            for x in home:
                for y in month:
                    for z in address:
                        t0 = (situated[x, z] & is_linked_with_1[y, z]).implies(investigated_in[x, y])
                        constraints_transitivity[0].append(t0)

                        t1 = (~situated[x, z] & is_linked_with_1[y, z]).implies(~investigated_in[x, y])
                        constraints_transitivity[1].append(t1)

                        t2 = (situated[x, z] & ~is_linked_with_1[y, z]).implies(~investigated_in[x, y])
                        constraints_transitivity[2].append(t2)

            for x in home:
                for y in month:
                    for z in ghost:
                        t3 = (haunted_by[x, z] & is_linked_with_2[y, z]).implies(investigated_in[x, y])
                        constraints_transitivity[3].append(t3)


                        t4 = (~haunted_by[x, z] & is_linked_with_2[y, z]).implies(~investigated_in[x, y])
                        constraints_transitivity[4].append(t4)

                        t5 = (haunted_by[x, z] & ~is_linked_with_2[y, z]).implies(~investigated_in[x, y])
                        constraints_transitivity[5].append(t5)



            for x in home:
                for y in address:
                    for z in ghost:
                        t6 = (haunted_by[x, z] & is_linked_with_3[y, z]).implies(situated[x, y])
                        constraints_transitivity[6].append(t6)

                        t7 = (~haunted_by[x, z] & is_linked_with_3[y, z]).implies(~situated[x, y])
                        constraints_transitivity[7].append(t7)

                        t8 = (haunted_by[x, z] & ~is_linked_with_3[y, z]).implies(~situated[x, y])
                        constraints_transitivity[8].append(t8)


            for x in month:
                for y in address:
                    for z in ghost:
                        t9 = (is_linked_with_2[x, z] & is_linked_with_3[y, z]).implies(is_linked_with_1[x, y])
                        constraints_transitivity[9].append(t9)

                        t10 = (~is_linked_with_2[x, z] & is_linked_with_3[y, z]).implies(~is_linked_with_1[x, y])
                        constraints_transitivity[10].append(t10)

                        t11 = (is_linked_with_2[x, z] & ~is_linked_with_3[y, z]).implies(~is_linked_with_1[x, y])
                        constraints_transitivity[11].append(t11)

            for set_constraint in constraints_transitivity:
                constraints.append(all(set_constraint))
                dict_constraints_clues[all(set_constraint)] = 'transitivity'
                dict_constraint_dv[all(set_constraint)] = extract_dv_from_expression(all(set_constraint),involved=[])
                dict_constraint_type[all(set_constraint)] = 'transitivity'

            # 0. The home visited in April was either Markmanor or the home haunted by Brunhilde
            c0a = []
            for q in home:
                subformula = any(
                    haunted_by[r, "brunhilde"]
                    for r in home if r == q
                )
                c0a.append(investigated_in[q, "4"] & (("markmanor" == q) | subformula))
            constraints.append(any(c0a))
            dict_constraint_dv[any(c0a)] = extract_dv_from_expression(any(c0a),involved=[])
            dict_constraint_type[any(c0a)] = 'clue'
            dict_constraints_clues[any(c0a)] = 'The home visited in April was either Markmanor or the home haunted by Brunhilde'

            #
            # # 1. Hughenden wasn't investigated in march
            c1a = ~ investigated_in["hughenden", "3"]
            facts.append(c1a)

            #
            # # 2. The home on Circle Drive was investigated sometime before Wolfenden
            c2a = []
            for a in home:
                for c in month:
                    for d in month:
                        if int(d) < int(c):
                            c2a.append(
                                situated[a, "circle_drive"] & investigated_in["wolfenden", c] & investigated_in[a, d])
            constraints.append(any(c2a))
            dict_constraint_dv[any(c2a)] = extract_dv_from_expression(any(c2a),involved=[])
            dict_constraint_type[any(c2a)] = 'clue'
            dict_constraints_clues[any(c2a)] = 'The home on Circle Drive was investigated sometime before Wolfenden'


            #
            # # 3. Of the building haunted by Lady Grey and the building haunted by Victor, one was Markmanor and the other was visited in January
            c3a = []
            for e in home:
                for f in home:
                    if not (e == f):
                        c3a.append(
                            haunted_by[e, "lady_grey"] & haunted_by[f, "victor"] & (
                                    (("markmanor" == e) & investigated_in[f, "1"]) | (
                                    ("markmanor" == f) & investigated_in[e, "1"]))
                        )
            constraints.append(any(c3a))
            dict_constraint_dv[any(c3a)] = extract_dv_from_expression(any(c3a),involved=[])
            dict_constraint_type[any(c3a)] = 'clue'
            dict_constraints_clues[any(c3a)] = 'Of the building haunted by Lady Grey and the building haunted by Victor, one was Markmanor and the other was visited in January'

            # # 4. The house haunted by Victor was visited 1 month after the house haunted by Lady Grey
            c4a = []
            for g in home:
                for h in month:
                    for i in home:
                        for j in month:
                            if int(j) == int(h) + 1:
                                c4a.append(
                                    haunted_by[g, "victor"] & haunted_by[i, "lady_grey"] & investigated_in[i, h] &
                                    investigated_in[g, j]
                                )
            constraints.append(any(c4a))
            dict_constraint_dv[any(c4a)] = extract_dv_from_expression(any(c4a),involved=[])
            dict_constraint_type[any(c4a)] = 'clue'
            dict_constraints_clues[any(c4a)] = 'The house haunted by Victor was visited 1 month after the house haunted by Lady Grey'

            # # 5. Of the home on Bird Road and Barnhill, one was visited in January and the other was haunted by Brunhilde
            c5a = []
            for k in home:
                if not k == "barnhill":
                    c5a.append(situated[k, "bird_road"] &
                               ((investigated_in[k, "1"] & haunted_by["barnhill", "brunhilde"]) |
                                (investigated_in["barnhill", "1"] & haunted_by[k, "brunhilde"])))

            constraints.append(any(c5a))
            dict_constraint_dv[any(c5a)] = extract_dv_from_expression(any(c5a),involved=[])
            dict_constraint_type[any(c5a)] = 'clue'
            dict_constraints_clues[any(c5a)] = 'Of the home on Bird Road and Barnhill, one was visited in January and the other was haunted by Brunhilde'

            # # 6. Markmanor was visited 1 month after the home on Grant Place
            c6a = []
            for l in month:
                for m in home:
                    for n in month:
                        if int(n) == int(l) + 1:
                            c6a.append(
                                situated[m, "grant_place"] & investigated_in[m, l] & investigated_in["markmanor", n])

            constraints.append(any(c6a))
            dict_constraint_dv[any(c6a)] = extract_dv_from_expression(any(c6a),involved=[])
            dict_constraint_type[any(c6a)] = 'clue'
            dict_constraints_clues[any(c6a)] = 'Markmanor was visited 1 month after the home on Grant Place'


            # # 7. The house visited in march wasn't located on Circle Drive
            c7a = []
            for o in home:
                c7a.append(investigated_in[o, "3"] & ~ situated[o, "circle_drive"])

            constraints.append(any(c7a))
            dict_constraint_dv[any(c7a)] = extract_dv_from_expression(any(c7a),involved=[])
            dict_constraint_type[any(c7a)] = 'clue'
            dict_constraints_clues[any(c7a)] = 'The house visited in march wasn\'t located on Circle Drive'

            # # 8. Hughenden wasn't haunted by Abigail
            c8a = ~ haunted_by["hughenden", "abigail"]
            facts.append(c8a)


            # # # 9. Wolfenden was haunted by Brunhilde
            c9a = haunted_by["wolfenden", "brunhilde"]
            facts.append(c9a)


            # # 10. The building visited in May wasn't located on Fifth Avenue
            c10a = []
            for p in home:
                c10a.append(investigated_in[p, "5"] & ~ situated[p, "fifth_avenue"])
            constraints.append(any(c10a))
            dict_constraint_dv[any(c10a)] = extract_dv_from_expression(any(c10a),involved=[])
            dict_constraint_type[any(c10a)] = 'clue'
            dict_constraints_clues[any(c7a)] = 'The building visited in May wasn\'t located on Fifth Avenue'

            top_level_facts = toplevel_list(facts, merge_and=False)
            explained = []
            for f in top_level_facts:
                if isinstance(f, NegBoolView):
                    explained.append(f._bv)
                else:
                    explained.append(f)
            # bvRels = {}
            for rel, relStr in zip(rels, ["investigated_in", "on", "haunted_by", "is_linked_with_1", "is_linked_with_2",
                                          "is_linked_with_3"]):

                # facts to explain
                for item in rel.df.values:
                    explainable_facts |= set(x for x in item if x not in set(explained))
            dict_adjacency = define_dict_adjacency(constraints, facts, explainable_facts, dict_constraint_type, dict_constraint_dv)
            return ([], constraints,facts,explainable_facts,dict_constraint_type,dict_adjacency,
                    rels_visualization,dict_constraints_clues)
        if self.type == 1:

            linkedin_connection = ["57", "59", "64", "68", "78"]
            person = ["opal", "neil", "rosie", "arnold", "georgia"]
            facebook_friend = ["120", "130", "140", "150", "160"]
            twitter_follower = ["589", "707", "715", "789", "809"]

            types = [linkedin_connection, person, facebook_friend, twitter_follower]

            n = len(types)
            m = len(types[0])
            assert all(len(types[i]) == m for i in range(n)), "all types should have equal length"

            # relations between types
            connected_with = Relation(person, linkedin_connection,name='connected')  # investigated_in(home, month)
            friend_of = Relation(person, facebook_friend,name='friend')  # on(home, type1)
            followed_by = Relation(person, twitter_follower,name='followed')  # haunted_by(home, type2)
            is_linked_with_1 = Relation(linkedin_connection, facebook_friend,name='first')  # is_linked_with_1(month, type1)
            is_linked_with_2 = Relation(linkedin_connection, twitter_follower,name='second')  # is_linked_with_2(month, type2)
            is_linked_with_3 = Relation(facebook_friend, twitter_follower,name='third')  # is_linked_with_3(type1, type2)

            rels = [connected_with, friend_of, followed_by, is_linked_with_1, is_linked_with_2, is_linked_with_3]
            rels_visualization = [[followed_by.df, friend_of.df, connected_with.df],
                                  [is_linked_with_2.df, is_linked_with_1.df],
                                  [is_linked_with_3.df]]

            # Bijectivity
            constraints_row,constraints_columns = self.buildBijectivity(rels)
            for cnstr in constraints_row:
                constraints.append(cnstr)
                dict_constraints_clues[cnstr] = 'bijectivity'
                dict_constraint_dv[cnstr] = extract_dv_from_expression(cnstr,involved=[])
                dict_constraint_type[cnstr] = 'bijectivity'
            for cnstr in constraints_columns:
                constraints.append(cnstr)
                dict_constraints_clues[cnstr] = 'bijectivity'
                dict_constraint_dv[cnstr] = extract_dv_from_expression(cnstr,involved=[])
                dict_constraint_type[cnstr] = 'bijectivity'


            # Transitivity
            constraints_transitivity = [[] for _ in range(12)]
            for x in person:
                for y in linkedin_connection:
                    for z in facebook_friend:

                        t0 = (friend_of[x, z] & is_linked_with_1[y, z]).implies(connected_with[x, y])
                        constraints_transitivity[0].append(t0)

                        t1 = (~friend_of[x, z] & is_linked_with_1[y, z]).implies(~connected_with[x, y])
                        constraints_transitivity[1].append(t1)

                        t2 = (friend_of[x, z] & ~is_linked_with_1[y, z]).implies(~connected_with[x, y])
                        constraints_transitivity[2].append(t2)

            for x in person:
                for y in linkedin_connection:
                    for z in twitter_follower:

                        t3 = (followed_by[x, z] & is_linked_with_2[y, z]).implies(connected_with[x, y])
                        constraints_transitivity[3].append(t3)

                        t4 = (~followed_by[x, z] & is_linked_with_2[y, z]).implies(~connected_with[x, y])
                        constraints_transitivity[4].append(t4)

                        t5 = (followed_by[x, z] & ~is_linked_with_2[y, z]).implies(~connected_with[x, y])
                        constraints_transitivity[5].append(t5)



            for x in person:
                for y in facebook_friend:
                    for z in twitter_follower:

                        t6 = (followed_by[x, z] & is_linked_with_3[y, z]).implies(friend_of[x, y])
                        constraints_transitivity[6].append(t6)

                        t7 = (~followed_by[x, z] & is_linked_with_3[y, z]).implies(~friend_of[x, y])
                        constraints_transitivity[7].append(t7)

                        t8 = (followed_by[x, z] & ~is_linked_with_3[y, z]).implies(~friend_of[x, y])
                        constraints_transitivity[8].append(t8)


            for x in linkedin_connection:
                for y in facebook_friend:
                    for z in twitter_follower:

                        t9 = (is_linked_with_2[x, z] & is_linked_with_3[y, z]).implies(is_linked_with_1[x, y])
                        constraints_transitivity[9].append(t9)

                        t10 = (~is_linked_with_2[x, z] & is_linked_with_3[y, z]).implies(~is_linked_with_1[x, y])
                        constraints_transitivity[10].append(t10)

                        t11 = (is_linked_with_2[x, z] & ~is_linked_with_3[y, z]).implies(~is_linked_with_1[x, y])
                        constraints_transitivity[11].append(t11)

            for set_constraint in constraints_transitivity:
                constraints.append(all(set_constraint))
                dict_constraints_clues[all(set_constraint)] = 'transitivity'
                dict_constraint_dv[all(set_constraint)] = extract_dv_from_expression(all(set_constraint),involved=[])
                dict_constraint_type[all(set_constraint)] = 'transitivity'

            # 0. The person followed by 809 Twitter followers, the person with 140 facebook friends and the person
            #    connected to 78 linkedin connections are three different people
            c0a = []
            for a in person:
                for b in person:
                    for c in person:
                        if not (a == b) and not (a == c) and not (b == c):
                            c0a.append(followed_by[a, "809"] & friend_of[b, "140"] & connected_with[c, "78"])

            constraints.append(any(c0a))
            dict_constraint_dv[any(c0a)] = extract_dv_from_expression(any(c0a),involved=[])
            dict_constraint_type[any(c0a)] = 'clue'
            dict_constraints_clues[any(c0a)] = 'The person followed by 809 Twitter followers, the person with 140 facebook friends and the person connected to 78 linkedin connections are three different people'

            #
            # 1. Of rosie and neil, one is connected to 68 linkedin connections and the other is followed by 789 twitter followers
            c1a = (connected_with["rosie", "68"] & followed_by["neil", "789"]) | (connected_with["neil", "68"] & followed_by["rosie", "789"])
            constraints.append(c1a)
            dict_constraint_dv[c1a] = extract_dv_from_expression(c1a)
            dict_constraint_type[c1a] = 'clue'
            dict_constraints_clues[c1a] = 'Of rosie and neil, one is connected to 68 linkedin connections and the other is followed by 789 twitter followers'


            # 2. The person connected to 57 linkedin connections has 10 facebook friends less than the person followed by 715 twitter followers
            c2a = []
            for d in person:
                for e in facebook_friend:
                    for f in person:
                        for g in facebook_friend:
                            if int(g) == int(e) - 10:
                                c2a.append(connected_with[d, "57"] & followed_by[f, "715"] & friend_of[f, e] &
                                           friend_of[d, g])
            constraints.append(any(c2a))
            dict_constraint_dv[any(c2a)] = extract_dv_from_expression(any(c2a),involved=[])
            dict_constraint_type[any(c2a)] = 'clue'
            dict_constraints_clues[any(c2a)] = 'The person connected to 57 linkedin connections has 10 facebook friends less than the person followed by 715 twitter followers'


            # 3. Arnold isn't followed by 589 twitter followers
            c3a = ~ followed_by["arnold", "589"]
            facts.append(c3a)

            # 4. The person followed by 809 twitter followers isn't connected to 68 linkedin connections
            c4a = []
            for h in person:
                c4a.append(followed_by[h, "809"] & ~ connected_with[h, "68"])
            constraints.append(any(c4a))
            dict_constraint_dv[any(c4a)] = extract_dv_from_expression(any(c4a),involved=[])
            dict_constraint_type[any(c4a)] = 'clue'
            dict_constraints_clues[any(c4a)] = 'The person followed by 809 twitter followers isn''t connected to 68 linkedin connections'

            # 5. Of the person connected to 57 linkedin connections and arnold, one has 140 facebook friends and the other is followed by 789 twitter followers
            c5a = []
            for i in person:
                if not (i == "arnold"):
                    c5a.append(connected_with[i, "57"] &
                               ((friend_of[i, "140"] & followed_by["arnold", "789"]) |
                                (friend_of["arnold", "140"] & followed_by[i, "789"])))

            constraints.append(any(c5a))
            dict_constraint_dv[any(c5a)] = extract_dv_from_expression(any(c5a),involved=[])
            dict_constraint_type[any(c5a)] = 'clue'
            dict_constraints_clues[any(c5a)] = 'Of the person connected to 57 linkedin connections and arnold, one has 140 facebook friends and the other is followed by 789 twitter followers'


            # 6. opal doesn't have 150 facebook friends
            c6a = ~friend_of["opal", "150"]
            facts.append(c6a)


            # 7. the person connected to 57 linkedin connections has 10 facebook friends less than georgia
            c7a = []
            for j in person:
                for k in facebook_friend:
                    for l in facebook_friend:
                        if int(l) == int(k) - 10:
                            c7a.append(connected_with[j, "57"] & friend_of["georgia", k] & friend_of[j, l])

            constraints.append(any(c7a))
            dict_constraint_dv[any(c7a)] = extract_dv_from_expression(any(c7a),involved=[])
            dict_constraint_type[any(c7a)] = 'clue'
            dict_constraints_clues[any(c7a)] = 'the person connected to 57 linkedin connections has 10 facebook friends less than georgia'

            # 8. The person with 130 facebook friends is either arnold or the person followed by 715 twitter followers
            c8a = []
            for p in person:
                c8a.append(friend_of[p, '130'] & Xor([p == 'arnold', followed_by[p, '715']]))

            constraints.append(any(c8a))
            dict_constraint_dv[any(c8a)] = extract_dv_from_expression(any(c8a), involved=[])
            dict_constraint_type[any(c8a)] = 'clue'
            dict_constraints_clues[any(c8a)] = 'The person with 130 facebook friends is either arnold or the person followed by 715 twitter followers'

            # 9. the person followed by 789 twitter followers has less facebook friends than rosie
            c9a = []
            for o in person:
                if person != 'rosie':
                    for p in facebook_friend:
                        for r in facebook_friend:
                            if int(r) < int(p):
                                c9a.append(followed_by[o, "789"] & friend_of["rosie", p] & friend_of[o, r])

            constraints.append(any(c9a))
            dict_constraint_dv[any(c9a)] = extract_dv_from_expression(any(c8a), involved=[])
            dict_constraint_type[any(c9a)] = 'clue'
            dict_constraints_clues[any(c9a)] = 'the person followed by 789 twitter followers has less facebook friends than rosie'

            # 10. opal doesn't have 150 facebook friends
            c10a = ~connected_with["opal", "64"]
            facts.append(c10a)



            top_level_facts = toplevel_list(facts, merge_and=False)
            explained = []
            for f in top_level_facts:
                if isinstance(f, NegBoolView):
                    explained.append(f._bv)
                else:
                    explained.append(f)
            for rel, relStr in zip(rels, ["investigated_in", "on", "haunted_by", "is_linked_with_1", "is_linked_with_2",
                                          "is_linked_with_3"]):

                # facts to explain
                for item in rel.df.values:
                    explainable_facts |= set(x for x in item if x not in set(explained))

            dict_adjacency = define_dict_adjacency(constraints, facts, explainable_facts,
                                                   dict_constraint_type, dict_constraint_dv)
            return ([], constraints, facts, explainable_facts, dict_constraint_type,
                    dict_adjacency,rels_visualization, dict_constraints_clues)

        return False

    def buildBijectivity(self,rels):
        constraints_row = []
        constraints_columns = []
        for rel in rels:
            bij_rows = []
            bij_columns = []
            for col_ids in rel.df:
                clause = self.exactly_one(rel[:, col_ids])
                bij_rows.append(clause)
            constraints_row.append(all(bij_rows))

            for (_, row) in rel.df.iterrows():
                clause = self.exactly_one(row)
                bij_columns.append(clause)
            constraints_columns.append(all(bij_columns))
        return constraints_row, constraints_columns

    def exactly_one(self,lst):
        clause = sum(el for el in lst) == 1
        return clause




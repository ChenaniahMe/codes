function  [Sample]=extract_features(Rows,Cols,T)

Sample = zeros(Rows, Cols, 41);
% CloudeFeatures=cell(SampleNum);
% HuynenFeatures=cell(SampleNum);
% Pauli=cell(SampleNum);
IncrementNum=zeros(1,15);
Span_all=zeros(Rows,Cols);
Tmatrix=T;
for ii = 1 : Rows
    for jj = 1 : Cols
        Span_all(ii,jj) = trace(Tmatrix{ii,jj});
    end
end
SpanMax5 = max(Span_all(:));
SpanMin5 = min(Span_all(:));
SpanMin5 = max(SpanMin5,eps);
for m=1:Rows
    %fprintf('???Complete%3d????????\n', m);
    fprintf('Complete the compution of samples features of %3d\n', m);
    for n=1:Cols
        % if label(m,n)~=0
        BB1=[1 0 1;1 0 -1;0 sqrt(2) 0]/sqrt(2);
        T8=Tmatrix{m,n};
        C8=BB1'*T8*BB1;
        TT12=sqrt(abs(T8(1,2)));
        angleTT12=angle(T8(1,2));
        TT13=sqrt(abs(T8(1,3)));
        angleTT13=angle(T8(1,3));
        TT23=sqrt(abs(T8(2,3)));
        angleTT23=angle(T8(2,3));
        
        CT12=sqrt(abs(C8(1,2)));
        angleCT12=angle(C8(1,2));
        CT13=sqrt(abs(C8(1,3)));
        angleCT13=angle(C8(1,3));
        CT23=sqrt(abs(C8(2,3)));
        angleCT23=angle(C8(2,3));
        %aa=[T11;T22;T33;C11;C22;C33;C12;angle12;C13;angle13;C23;angle23];
        aa1=[TT12;angleTT12;TT13;angleTT13;TT23;angleTT23;CT12;angleCT12;CT13;angleCT13;CT23;angleCT23];
        bb1=norm(aa1);
        
        TT12=TT12/bb1;
        angleTT12=angleTT12/bb1;
        TT13=TT13/bb1;
        angleTT13=angleTT13/bb1;
        TT23=TT23/bb1;
        angleTT23=angleTT23/bb1;
        
        CT12=CT12/bb1;
        angleCT12=angleCT12/bb1;
        CT13=CT13/bb1;
        angleCT13=angleCT13/bb1;
        CT23=CT23/bb1;
        angleCT23=angleCT23/bb1;
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        % Cloude??
        %T=[T11(m,n) T12(m,n) T13(m,n);T12(m,n)' T22(m,n) T23(m,n);T13(m,n)' T23(m,n)' T33(m,n)];
        B2=[1 0 1;1 0 -1;0 sqrt(2) 0]/sqrt(2);
        %            C{m,n}=B'*T{m,n}*B; %???????
        [V2,D2]=eigs(Tmatrix{m,n});  %[V,D] = eigs(A,…)   %D?6??????????V????????????
        lammda2=diag(D2); %diag???????
        
        p2=lammda2/sum(lammda2);
        
        if lammda2(3)<0;
            lammda2(3)=0;
            lammda2(2)=lammda2(2)-lammda2(3);
            lammda2(1)=lammda2(1)-lammda2(3);
        end
        %H(m,n)=(-sum( p.*log(p)/log(3) )); % H
        
        H2=-(p2(1)*log(p2(1))/log(3)+p2(2)*log(p2(2))/log(3)+p2(3)*log(p2(3))/log(3));
        %alpha=acos(abs(V(1,:)))*180/pi;
        alpha2=acos(abs(V2(1,:)));
        alpha02=alpha2*p2; % alpha
        A2=(lammda2(2)-lammda2(3))/(lammda2(2)+lammda2(3));
        
        
        %H(m,n)=H(m,n)/100;
        %alpha0(m,n)=alpha0(m,n)/100000;
        %A=A/1000;
        % lammda(1)=lammda(1)/10;
        % CloudeFeatures{n,m}={H(m,n),alpha0(m,n),A,lammda(1),lammda(2),lammda(3)};
        aaa2=[H2;alpha02;A2;lammda2(1);lammda2(2);lammda2(3)];
        bbb2=norm(aaa2);
        HH2=H2/bbb2;
        alpha02=alpha02/bbb2;
        A2=A2/bbb2;
        lammda2(1)=lammda2(1)/bbb2;
        lammda2(2)=lammda2(2)/bbb2;
        lammda2(3)=lammda2(3)/bbb2;
        
        
        % CloudeFeatures{m,n}={H(m,n),alpha0(m,n),A,lammda(1),lammda(2),lammda(3)};
        %CloudeFeatures{n,m}={H(m,n),alpha0(m,n),A};
        %        I1(m,n,:)=clr(:,z(m,n));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%Huynen??%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        t3=Tmatrix{m,n};
        A0a3=t3(1,1)/2;
        Cc3=real(t3(1,2));
        Dd3=imag(t3(1,2));
        Hh3=real(t3(1,3));
        Gg3=imag(t3(1,3));
        B0b3=t3(2,2);%B0+B
        Ee3=real(t3(2,3));
        Ff3=imag(t3(2,3));
        Bb3=t3(3,3);%B0-B
        
        ccc3=[A0a3;Cc3;Dd3;Hh3;Gg3;B0b3;Ee3;Ff3;Bb3];
        ddd3=norm(ccc3);
        A0a3=A0a3/ddd3;
        Cc3=Cc3/ddd3;
        Dd3=Dd3/ddd3;
        Hh3=Hh3/ddd3;
        Gg3=Gg3/ddd3;
        B0b3=B0b3/ddd3;
        Ee3=Ee3/ddd3;
        Ff3=Ff3/ddd3;
        Bb3=Bb3/ddd3;
        % HuynenFeatures{m,n}={A0,C,D,H,G,B0,E,F,B};
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%Pauli??%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tt4=Tmatrix{m,n};
        tt4=abs(tt4);
        nn4=sqrt(2);
        
        
        
        aa4=tt4(1,1);
        
        bb4=tt4(2,2);
        
        cc4=tt4(3,3);
        
        eee4=[aa4,bb4,cc4];
        fff4=norm(eee4);
        aa4=aa4/fff4;
        bb4=bb4/fff4;
        cc4=cc4/fff4;
        %  Pauli{m,n}={aa,bb,cc};
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        t5=Tmatrix{m,n};
        if t5(2,2)>t5(3,3)
            angle15=1/4*atan(2*real(t5(2,3))/(t5(2,2)-t5(3,3)));
        elseif t5(2,2)<t5(3,3)
            angle15=1/4*(pi+atan(2*real(t5(2,3))/(t5(2,2)-t5(3,3))));
        else
            angle15 = 0;   % %lhy
        end
        % angle=0.25*atan(2*real(t(2,3))/(t(2,2)-t(3,3))); %?????angle
        %R=[1 0 0;0 cos(2*angle) sin(2*angle);0 -sin(2*angle) cos(2*angle)];
        %T=R*t*(R'); %????????
        Tt5(1,1)=t5(1,1);
        Tt5(1,2)=t5(1,2)*cos(2*angle15)+t5(1,3)*sin(2*angle15);
        Tt5(1,3)=-1*t5(1,2)*sin(2*angle15)+t5(1,3)*cos(2*angle15);
        Tt5(2,1)=Tt5(1,2)';
        Tt5(2,2)=t5(2,2)*(cos(2*angle15)^2)+t5(3,3)*(sin(2*angle15)^2)+real(t5(2,3))*sin(4*angle15);
        Tt5(2,3)=imag(t5(2,3))*1i;
        Tt5(3,1)=Tt5(1,3)';
        Tt5(3,2)=imag(t5(2,3))*1i;
        Tt5(3,3)=t5(3,3)*(cos(2*angle15)^2)+t5(2,2)*(sin(2*angle15)^2)-real(t5(2,3))*sin(4*angle15);
        
        Pc5=2*abs(imag(Tt5(2,3)));%??Pc
        rtio5=10*log((Tt5(1,1)+Tt5(2,2)-2*real(Tt5(1,2)))/(Tt5(1,1)+Tt5(2,2)+2*real(Tt5(1,2))));%Volume scttering power judgment  (-2db 2db)
        if Tt5(1,1)>(Tt5(2,2)-0.5*Pc5)      %???surfce  scttering  ?? Double bonuce scttering
            
            if rtio5>2
                Pv5=(15/4)*Tt5(3,3)-((15/8)*Pc5);
            elseif rtio5<-2
                Pv5=(15/4)*Tt5(3,3)-((15/8)*Pc5);
            else
                Pv5=4*Tt5(3,3)-(2*Pc5);
            end
            
            if Pv5<0
                Pc5=0;
                T115 = Tt5(1,1);
                T12_re5 = real(Tt5(1,2));
                T12_im5 = imag(Tt5(1,2));
                T13_im5 = imag(Tt5(1,3));
                T225 = Tt5(2,2);
                T335 = Tt5(3,3);
                C115 = (T115 + 2 * T12_re5 + T225) / 2;
                C13_re5 = (T115 - T225) / 2;
                C13_im5 = -T12_im5;
                C225 = T335;
                C335 = (T115 - 2 * T12_re5 + T225) / 2;
                HHHH5 = C115;
                HVHV5 = C225 / 2;
                VVVV5 = C335;
                HHVVre5 = C13_re5;
                HHVVim5 = C13_im5;
                
                if (rtio5 <= -2)
                    FV5 = 15 * HVHV5 / 4;
                    HHHH5 = HHHH5 - 8 * FV5 / 15;
                    VVVV5 = VVVV5 - 3 * FV5 / 15;
                    HHVVre5 = HHVVre5 - 2 * FV5/ 15;
                elseif (rtio5 > 2)
                    FV5 = 15 * HVHV5 / 4;
                    HHHH5 = HHHH5 - 3 * FV5 / 15;
                    VVVV5 = VVVV5 - 8 * FV5 / 15;
                    HHVVre5 = HHVVre5 - 2 * FV5 / 15;
                else %((rtio > -2.)&&(rtio <= 2.))
                    FV5 = 8 * HVHV5 / 2;
                    HHHH5 = HHHH5 - 3 * FV5 / 8;
                    VVVV5 = VVVV5 - 3 * FV5 / 8;
                    HHVVre5 = HHVVre5 - 1 * FV5 / 8;
                end
                %Cse 1: Volume Sctter > Totl
                if ((HHHH5 <= eps) || (VVVV5 <= eps))
                    FD5 = 0;
                    FS5 = 0;
                    if (rtio5 > -2)&&(rtio5 <= 2)
                        FV5 = (HHHH5 + 3 * FV5 / 8) + HVHV5 + (VVVV5 + 3 * FV5 / 8);
                    elseif (rtio5 <= -2)
                        FV5 = (HHHH5 + 8 * FV5 / 15) + HVHV5 + (VVVV5 + 3 * FV5 / 15);
                    else % (rtio > 2.)
                        FV5 = (HHHH5 + 3 * FV5 / 15) + HVHV5 + (VVVV5 + 8 * FV5 / 15);
                    end
                    Ps5=FS5;
                    Pd5=FD5;
                    Pv5=FV5;
                    Pv5 = max(min(Pv5,SpanMax5),SpanMin5);
                else %Dt conditionning for non realizble ShhSvv* term
                    if ((HHVVre5 * HHVVre5 + HHVVim5 * HHVVim5) > HHHH5 * VVVV5)
                        rtemp5 = HHVVre5 * HHVVre5 + HHVVim5 * HHVVim5;
                        HHVVre5 = HHVVre5 * sqrt(HHHH5 * VVVV5 / rtemp5);
                        HHVVim5 = HHVVim5 * sqrt(HHHH5 * VVVV5 / rtemp5);
                    end
                    %Odd Bounce*/
                    if (HHVVre5 >= 0)
                        LPre5 = -1;
                        LPim5 = 0;
                        FD5 = (HHHH5 * VVVV5 - HHVVre5 * HHVVre5 - HHVVim5 * HHVVim5) / (HHHH5 + VVVV5 + 2 * HHVVre5);
                        FS5 = VVVV5 - FD5;
                        BETre5 = (FD5 + HHVVre5) / FS5;
                        BETim5 = HHVVim5 / FS5;
                        %Even Bounce*/
                    else %(HHVVre < 0.)
                        BETre5 = 1;
                        BETim5 = 0;
                        FS5 = (HHHH5 * VVVV5 - HHVVre5 * HHVVre5 - HHVVim5 * HHVVim5) / (HHHH5 + VVVV5 - 2 * HHVVre5);
                        FD5 = VVVV5 - FS5;
                        LPre5 = (HHVVre5 - FS5) / FD5;
                        LPim5 = HHVVim5 / FD5;
                    end
                    Ps5 = FS5 * (1 + BETre5 * BETre5 + BETim5 * BETim5);
                    Ps5 = max(min(Ps5,SpanMax5),SpanMin5);
                    Pd5 = FD5 * (1 + LPre5 * LPre5 + LPim5 * LPim5);
                    Pd5 = max(min(Pd5,SpanMax5),SpanMin5);
                    Pv5 = FV5;
                    Pv5 = max(min(Pv5,SpanMax5),SpanMin5);
                end
            else
                if rtio5>2             %ratio?????S,D,C???
                    S5=Tt5(1,1)-0.5*Pv5;
                    D5=Tt5(2,2)-(7/30)*Pv5-0.5*Pc5;
                    C5=Tt5(1,2)+(1/6)*Pv5;
                elseif rtio5<-2
                    S5=Tt5(1,1)-0.5*Pv5;
                    D5=Tt5(2,2)-(7/30)*Pv5-0.5*Pc5;
                    C5=Tt5(1,2)-(1/6)*Pv5;
                else
                    S5=Tt5(1,1)-0.5*Pv5;
                    D5=Tt5(2,2)-Tt5(3,3);
                    C5=Tt5(1,2);
                end
                TP5=Tt5(1,1)+Tt5(2,2)+Tt5(3,3);%?????TP
                
                if Pv5+Pc5>TP5   %??(Pv+PC)?TP???
                    Ps5=0;
                    Pd5=0;
                    Pv5=TP5-Pc5;
                else
                    C05=Tt5(1,1)-Tt5(2,2)-Tt5(3,3)+Pc5;  %??C0
                    if C05>0                      %??C0???
                        Ps5=S5+C5*C5'/S5;
                        Pd5=D5-C5*C5'/S5;
                        
                    else
                        Pd5=D5+C5*C5'/D5;
                        Ps5=S5-C5*C5'/D5;
                    end
                    if Ps5>0
                        if Pd5>0
                            %TP=Ps+Pd+Pv+Pc;
                        else
                            Pd5=0;
                            Ps5=TP5-Pv5-Pc5;
                        end
                    else
                        if Pd5>0
                            Ps5=0;
                            Pd5=TP5-Pv5-Pc5;
                        else
                            Ps5=0;
                            Pd5=0;
                            Pv5=TP5-Pc5;
                        end
                    end
                end
            end
        else                % Double bonuce scttering???
            Pv5=(15/8)*Tt5(3,3)-((15/16)*Pc5);
            TP5=Tt5(1,1)+Tt5(2,2)+Tt5(3,3);%?????TP
            if Pv5<0
                Pc5=0;
                T115 = t5(1,1);
                T12_re5 = real(t5(1,2));
                T12_im5 = imag(t5(1,2));
                T13_im5= imag(t5(1,3));
                T225 = t5(2,2);
                T335 = t5(3,3);
                C115 = (T115 + 2 * T12_re5 + T225) / 2;
                C13_re5 = (T115 - T225) / 2;
                C13_im5 = -T12_im5;
                C225 = T335;
                C335 = (T115 - 2 * T12_re5 + T225) / 2;
                HHHH5 = C115;
                HVHV5 = C225 / 2;
                VVVV5 = C335;
                HHVVre5 = C13_re5;
                HHVVim5 = C13_im5;
                
                
                if (rtio5 <= -2)    %%%%%%%%%%----------------------------bug
                    FV5 = 15 * HVHV5 / 4;
                    HHHH5 = HHHH5 - 8 * FV5 / 15;
                    VVVV5 = VVVV5 - 3 * FV5 / 15;
                    HHVVre5 = HHVVre5 - 2 * FV5 / 15;
                elseif (rtio5 > 2)
                    FV5 = 15 * HVHV5 / 4;
                    HHHH5 = HHHH5 - 3 * FV5 / 15;
                    VVVV5 = VVVV5 - 8 * FV5 / 15;
                    HHVVre5 = HHVVre5 - 2 * FV5 / 15;
                else %((rtio > -2.)&&(rtio <= 2.))
                    FV5 = 8 * HVHV5 / 2;
                    HHHH5 = HHHH5 - 3 * FV5 / 8;
                    VVVV5 = VVVV5 - 3 * FV5 / 8;
                    HHVVre5 = HHVVre5 - 1 * FV5 / 8;
                end
                %Cse 1: Volume Sctter > Totl
                if ((HHHH5 <= eps) || (VVVV5 <= eps))
                    FD5 = 0;
                    FS5 = 0;
                    if (rtio5 > -2)&&(rtio5 <= 2)
                        FV5 = (HHHH5 + 3 * FV5 / 8) + HVHV5 + (VVVV5 + 3 * FV5/ 8);
                    elseif (rtio5 <= -2)
                        FV5 = (HHHH5 + 8 * FV5 / 15) + HVHV5 + (VVVV5 + 3 * FV5 / 15);
                    else % (rtio > 2.)
                        FV5 = (HHHH5 + 3 * FV5 / 15) + HVHV5 + (VVVV5 + 8 * FV5 / 15);
                    end
                    Ps5=FS5;
                    Pd5=FD5;
                    Pv5=FV5;
                    Pv5 = max(min(Pv5,SpanMax5),SpanMin5);
                else %Dt conditionning for non realizble ShhSvv* term
                    if ((HHVVre5 * HHVVre5 + HHVVim5 * HHVVim5) > HHHH5 * VVVV5)
                        rtemp5 = HHVVre5 * HHVVre5 + HHVVim5 * HHVVim5;
                        HHVVre5 = HHVVre5 * sqrt(HHHH5 * VVVV5 / rtemp5);
                        HHVVim5 = HHVVim5 * sqrt(HHHH5 * VVVV5 / rtemp5);
                    end
                    %Odd Bounce*/
                    if (HHVVre5 >= 0)
                        LPre5 = -1;
                        LPim5 = 0;
                        FD5 = (HHHH5 * VVVV5 - HHVVre5 * HHVVre5 - HHVVim5 * HHVVim5) / (HHHH5 + VVVV5 + 2 * HHVVre5);
                        FS5 = VVVV5 - FD5;
                        BETre5 = (FD5 + HHVVre5) / FS5;
                        BETim5 = HHVVim5 / FS5;
                        %Even Bounce*/
                    else %(HHVVre < 0.)
                        BETre5 = 1;
                        BETim5 = 0;
                        FS5 = (HHHH5 * VVVV5 - HHVVre5 * HHVVre5 - HHVVim5* HHVVim5) / (HHHH5 + VVVV5 - 2 * HHVVre5);
                        FD5 = VVVV5 - FS5;
                        LPre5 = (HHVVre5 - FS5) / FD5;
                        LPim5 = HHVVim5 / FD5;
                    end
                    Ps5 = FS5 * (1 + BETre5 * BETre5 + BETim5 * BETim5);
                    Ps5 = max(min(Ps5,SpanMax5),SpanMin5);
                    Pd5 = FD5 * (1 + LPre5 * LPre5 + LPim5 * LPim5);
                    Pd5 = max(min(Pd5,SpanMax5),SpanMin5);
                    Pv5 = FV5;
                    Pv5 = max(min(Pv5,SpanMax5),SpanMin5);
                    
                end
                
            else
                S5=Tt5(1,1);
                D5=Tt5(2,2)-(7/15)*Pv5-0.5*Pc5;
                C5=Tt5(1,2);
                
                Pd5=D5+C5*C5'/D5;
                Ps5=S5-C5*C5'/D5;
                %??Ps,Pd?Pv?Pc?????
                % if (Pv+Pc)>TP
                %          Ps=0;
                %          Pd=0;
                %          Pv=TP-Pc;
                % else
                if Ps5>0
                    if Pd5>0
                        %TP=Ps+Pd+Pv+Pc;
                    else
                        Pd5=0;
                        Ps5=TP5-Pv5-Pc5;
                    end
                else
                    if Pd5>0
                        Ps5=0;
                        Pd5=TP5-Pv5-Pc5;
                    else
                        Ps5=0;
                        Pd5=0;
                        Pv5=TP5-Pc5;
                    end
                end
                % end
            end
        end
        psn5=Ps5;
        pdn5=Pd5;
        pvn5=Pv5;
        phn5=Pc5;
        aaa5=[Ps5;Pd5;Pv5;Pc5];
        bbb5=norm(aaa5);
        Pss=Ps5/bbb5;
        Pdd=Pd5/bbb5;
        Pvv=Pv5/bbb5;
        Pcc=Pc5/bbb5;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        nnnn6=[1 0 1;1 0 -1;0 sqrt(2) 0];
        N6=(1/sqrt(2))*nnnn6;
        TTTT6=Tmatrix{m,n};
        C6=inv(N6)*TTTT6*inv(N6');
        % C=TT;
        % CC(:,:,(ii-1)*width+jj)=C;
        R6=10*log(C6(3,3)/C6(1,1));
        RR6=R6;
        fv6=C6(2,2)*3/2;
        if real(C6(1,3))>=0
            if (imag(C6(1,3))==0)
                a6=-1;
                b6=(C6(1,1)+C6(1,3)-4*fv6/3)/(C6(1,3)+C6(3,3)-4*fv6/3);
                if (b6==-1)
                    fs6=0;
                else
                    fs6=(C6(1,3)+C6(3,3))/(b6+1);
                end
                if fs6<0
                    fs6=0;
                end
                fd6=C6(3,3)-fs6-fv6;
                if fd6<0
                    fd6=0;
                end
                
            else
                a6=-1;
                d6=(C6(1,1)-C6(3,3))/imag(C6(1,3));
                e6=(real(C6(1,3))+C6(3,3)-4*fv6/3)/imag(C6(1,3));
                %     b=(((d+2*e)/(e^2+1))*e-1)+((d+2*e)/(e^2+1))*i;
                b6=complex(((d6+2*e6)/(e6^2+1))*e6-1,(d6+2*e6)/(e6^2+1));
                if ((abs(b6))^2==1)
                    fs6=0;
                else
                    fs6=(C6(1,1)-C6(3,3))/((abs(b6))^2-1);
                    if fs6<0
                        fs6=0;
                    end
                end
                fd6=C6(3,3)-fs6-fv6;
                if fd6<0
                    fd6=0;
                end
                
            end
            
        else
            if (imag(C6(1,3))==0)
                b6=1;
                a6=(C6(1,1)-C6(1,3)-2*fv6/3)/(C6(1,3)-C6(3,3)+2*fv6/3);
                if (a6==1)
                    fd6=0;
                else
                    fd6=(C6(1,1)-C6(1,3)-2*fv6/3)/(a6-1);
                end
                if fd6<0
                    fd6=0;
                end
                fs6=C6(3,3)-fd6-fv6;
                if fs6<0
                    fs6=0;
                end
                
            else
                b6=1;
                d6=(real(C6(1,3))-C6(3,3))/imag(C6(1,3));
                e6=(C6(1,1)-(C6(3,3)))/imag(C6(1,3));
                a6=(((e6-2*d6)/(d6^2+1))*d6+1)+((e6-2*d6)/(d6^2+1))*i;
                if ((abs(a6))^2==1)
                    fd6=0;
                else
                    fd6=(C6(1,1)-C6(3,3))/((abs(a6))^2-1);
                end
                if fd6<0
                    fd6=0;
                end
                fs6=C6(3,3)-fd6-fv6;
                if fs6<0
                    fs6=0;
                end
                
            end
            
        end
        
        ps6=fs6*(1+(abs(b6))^2);
        pd6=fd6*(1+(abs(a6))^2);
        pv6=8*fv6/3;
        A6=a6;
        B6=b6;
        Fs6=fs6;
        Fv6=fv6;
        Fd6=fd6;
        Ps6=ps6;
        Pv6=pv6;
        Pd6=pd6;
        span6=C6(1,1)+C6(2,2)+C6(3,3);
        aa6=[ps6;pd6;pv6;span6];
        bb6=norm(aa6);
        pss6=ps6/bb6;
        pdd6=pd6/bb6;
        pvv6=pv6/bb6;
        spann6=span6/bb6;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %  Space(:,k1+IncrementNum(label(m,n)))=[m,n];
        Sample(m, n, :)=[A0a3,Cc3,Dd3,Hh3,Gg3,B0b3,Ee3,Ff3,Bb3,Pss,Pdd,Pvv,Pcc,HH2,alpha02,A2,lammda2(1),lammda2(2),lammda2(3),aa4,bb4,cc4,...
            pss6,pdd6,pvv6,fs6,fd6,fv6,spann6,TT12,angleTT12,TT13,angleTT13,TT23,angleTT23,CT12,angleCT12,CT13,angleCT13,CT23,angleCT23];
    end
end
end